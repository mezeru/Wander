import asyncio
import json
import logging
import random
from typing import Any

from langgraph.errors import GraphInterrupt
from langchain_core.messages import AIMessage, ToolMessage

import httpx
import ollama
from mcp import MCPError

from ..schemas import ResearchWorkerState
from ..specialist import distinct_places, tool_places
from .context import TripNodeContext


logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 8.0


def _transient_http_status(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response is not None:
            return response.status_code == 429 or response.status_code >= 500
    return False


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, ollama.RequestError):
        return True
    if isinstance(exc, ollama.ResponseError):
        status = getattr(exc, "status_code", None)
        return status == 408 or status == 429 or (isinstance(status, int) and status >= 500)
    if isinstance(exc, MCPError):
        code = getattr(exc, "error", None)
        code = getattr(code, "code", None) if code is not None else getattr(exc, "code", None)
        if code in {-32600, -32601, -32602}:
            return False
        return True
    return _transient_http_status(exc)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _distinct_report(
    report: dict[str, Any], key: str, live: bool,
) -> dict[str, Any]:
    records = distinct_places(report[key])
    for record in records:
        if key == "hotels":
            record["availability_verified"] = False
            if live:
                record.update(nightly_price=None, nightly_rate=None, currency=None)
        elif live:
            record.update(price=None, estimated_duration_hours=None)
    limitations = report.get("limitations")
    return {
        key: records,
        "limitations": (
            [value.strip() for value in limitations if isinstance(value, str) and value.strip()]
            if isinstance(limitations, list) else []
        ),
    }


def _extract_response(result: Any, agent: str, live: bool) -> tuple[str, Any, dict[str, Any]]:
    if isinstance(result, dict):
        explicit_status = result.get("status")
        if explicit_status == "no_data":
            return "no_data", None, {}

        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            successful = [message for message in tool_messages if message.status != "error"]
            counts = [len(places) for message in successful if (places := tool_places(message)) is not None]
            metadata = {
                "tool_messages": len(tool_messages),
                "tool_errors": len(tool_messages) - len(successful),
                "places_returned": sum(counts) if counts else None,
            }
            if tool_messages and not successful:
                return "failed", None, metadata
            if len(counts) == len(successful) and counts and not any(counts):
                return "no_data", None, {**metadata, "summarized_places": 0}

            last = messages[-1]
            if isinstance(last, ToolMessage):
                return "no_data", None, {**metadata, "summarized_places": 0}
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            text = _content_text(content)
            if text:
                if text.startswith("{"):
                    try:
                        report = json.loads(text)
                    except json.JSONDecodeError:
                        report = None
                    key = "attractions" if agent == "activities" else "hotels"
                    if isinstance(report, dict) and isinstance(report.get(key), list):
                        summary = _distinct_report(report, key, live)
                        count = len(summary[key])
                        status = "completed" if count else "no_data"
                        return status, summary, {**metadata, "summarized_places": count}
                return "completed", text, metadata

        text = _content_text(result.get("result"))
        if text:
            return "completed", text, {}
        return "no_data", None, {}
    text = _content_text(result)
    if text:
        return "completed", text, {}
    return "no_data", None, {}


def create_research_worker_node(context: TripNodeContext):
    async def research_worker(
        state: ResearchWorkerState,
    ) -> dict[str, list[dict[str, Any]]]:
        agent_name = state["agent"]
        job_id = state["job_id"]
        destination = state["destination"]

        if agent_name == "activities":
            graph = context.activities_graph
            tool_name = state.get("tool_name", "search_attractions")
        elif agent_name == "accommodation":
            graph = context.accommodation_graph
            tool_name = state.get("tool_name", "search_hotels")
        else:
            return {
                "research_results": [{
                    "job_id": job_id,
                    "agent": agent_name,
                    "destination": destination,
                    "status": "failed",
                    "result": None,
                    "error": f"Unknown research agent: {agent_name}",
                    "attempts": 1,
                }]
            }

        input_state = {
            "messages": [{
                "role": "user",
                "content": (
                    f"{state['task']}\n\n"
                    f"Validated search arguments: {json.dumps(state['arguments'])}\n"
                    f"Use the relevant preferences from this request: {state.get('user_request', '')}"
                ),
            }, AIMessage(content="", tool_calls=[{
                "name": tool_name, "args": state["arguments"], "id": job_id,
            }])],
        }

        for attempt in range(1, MAX_ATTEMPTS + 1):
            logger.info(
                "Research attempt job_id=%s agent=%s destination=%s attempt=%s/%s",
                job_id, agent_name, destination, attempt, MAX_ATTEMPTS,
            )
            try:
                logger.debug("Research tool job=%s tool=%s arguments=%s",
                             job_id, tool_name, state["arguments"])
                result = await graph.ainvoke(input_state)
                status, summary, metadata = _extract_response(
                    result, agent_name, getattr(context, "live_providers", False),
                )
                outcome = {
                    "job_id": job_id,
                    "agent": agent_name,
                    "destination": destination,
                    "status": status,
                    "result": summary if status == "completed" else None,
                    "error": "The MCP tool reported an error." if status == "failed" else None,
                    "attempts": attempt,
                }
                if metadata:
                    outcome["metadata"] = metadata
                logger.info("Research finished job=%s destination=%s status=%s", job_id, destination, status)
                logger.debug(
                    "Research tool outcome job=%s tool=%s reported_errors=%s returned=%s summarized=%s",
                    job_id, tool_name, metadata.get("tool_errors"), metadata.get("places_returned"),
                    metadata.get("summarized_places"),
                )
                return {"research_results": [outcome]}
            except (GraphInterrupt, asyncio.CancelledError):
                raise
            except Exception as exc:
                logger.info("Research attempt failed job=%s destination=%s error_type=%s",
                            job_id, destination, type(exc).__name__)
                if not _is_retryable(exc):
                    return {
                        "research_results": [{
                            "job_id": job_id,
                            "agent": agent_name,
                            "destination": destination,
                            "status": "failed",
                            "result": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempts": attempt,
                        }]
                    }
                if attempt == MAX_ATTEMPTS:
                    return {
                        "research_results": [{
                            "job_id": job_id,
                            "agent": agent_name,
                            "destination": destination,
                            "status": "failed",
                            "result": None,
                            "error": "Research service unavailable after retries",
                            "attempts": MAX_ATTEMPTS,
                        }]
                    }
                delay = min(2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
                delay *= 0.75 + random.random() * 0.5
                await asyncio.sleep(delay)

    return research_worker
