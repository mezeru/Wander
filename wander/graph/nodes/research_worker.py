import asyncio
import logging
import random
from typing import Any

from langgraph.errors import GraphInterrupt

import httpx
import ollama
from mcp import MCPError

from ..schemas import ResearchWorkerState
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


def _extract_response(result: Any) -> tuple[str, str, Any]:
    if isinstance(result, dict):
        explicit_status = result.get("status")
        if explicit_status == "no_data":
            return "no_data", "", result

        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            text = _content_text(content)
            if text:
                tool_metadata = [
                    message for message in messages
                    if getattr(message, "type", None) == "tool"
                    or isinstance(message, dict) and message.get("type") == "tool"
                ]
                return "completed", text, {"tool_messages": len(tool_metadata)}

        text = _content_text(result.get("result"))
        if text:
            return "completed", text, None
        if result == {} or result is None:
            return "no_data", "", result
        return "no_data", "", result

    if isinstance(result, (list, tuple)) and not result:
        return "no_data", "", result
    text = _content_text(result)
    if text:
        return "completed", text, None
    return "no_data", "", result


def create_research_worker_node(context: TripNodeContext):
    async def research_worker(
        state: ResearchWorkerState,
    ) -> dict[str, list[dict[str, Any]]]:
        agent_name = state["agent"]
        job_id = state["job_id"]
        destination = state["destination"]

        if agent_name == "activities":
            graph = context.activities_graph
        elif agent_name == "accommodation":
            graph = context.accommodation_graph
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
                    "Research arguments:\n"
                    f"{state['arguments']}"
                ),
            }]
        }

        for attempt in range(1, MAX_ATTEMPTS + 1):
            logger.info(
                "Research attempt job_id=%s agent=%s destination=%s attempt=%s/%s",
                job_id, agent_name, destination, attempt, MAX_ATTEMPTS,
            )
            try:
                result = await graph.ainvoke(input_state)
                status, text, metadata = _extract_response(result)
                outcome = {
                    "job_id": job_id,
                    "agent": agent_name,
                    "destination": destination,
                    "status": status,
                    "result": text if status == "completed" else None,
                    "error": None,
                    "attempts": attempt,
                }
                if metadata is not None:
                    outcome["metadata"] = metadata
                return {"research_results": [outcome]}
            except (GraphInterrupt, asyncio.CancelledError):
                raise
            except Exception as exc:
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
