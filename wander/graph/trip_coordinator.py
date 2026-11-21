from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..mcp_runtime import server_parameters
from ..providers.config import travel_settings
from .nodes import TripNodeContext, build_nodes
from .nodes.routes import (
    MAX_CLARIFICATION_ROUNDS,
    MAX_PLAN_ATTEMPTS,
    route_requirements,
    route_research_completeness,
    route_review,
    dispatch_research
)
from .schemas import (
    ResearchWorkerState,
    TripIntent,
    TripState,
    ValidationResult,
)
from .validation import (
    clean_task,
    dedupe_issues,
    explicit_nightly_budget,
    issue,
    missing,
    normalize_value,
    parse_date,
    parse_number,
    validate_arguments,
)


MODEL_NAME = "qwen3:4b"
DEMO_REQUEST = (
    "Plan a 3-day trip to Kyoto, Japan. Include sightseeing and a hotel. "
    "Hotel check-in: 2027-04-10. Hotel check-out: 2027-04-12. "
    "My nightly hotel budget is $150."
)
logger = logging.getLogger("wander.graph.trip_coordinator")

__all__ = [
    "MAX_CLARIFICATION_ROUNDS",
    "MAX_PLAN_ATTEMPTS",
    "MODEL_NAME",
    "ResearchWorkerState",
    "TripIntent",
    "TripState",
    "ValidationResult",
    "clean_task",
    "dedupe_issues",
    "explicit_nightly_budget",
    "issue",
    "main",
    "missing",
    "normalize_value",
    "parse_date",
    "parse_number",
    "tool_schema",
    "validate_arguments",
]



def tool_schema(tool: Any) -> dict[str, Any]:
    """Read the LangChain schema adapted from MCP discovery."""
    schema = tool.tool_call_schema
    return schema if isinstance(schema, dict) else schema.model_json_schema()


def build_workflow(node_context: TripNodeContext):
    workflow = StateGraph(TripState)
    for node_name, node in build_nodes(node_context).items():
        workflow.add_node(node_name, node)

    workflow.add_edge(START, "extract_trip_intent")
    workflow.add_edge("extract_trip_intent", "requirements_check")
    workflow.add_edge("create_research_jobs", "requirements_check")
    workflow.add_conditional_edges(
        "requirements_check",
        route_requirements,
        {
            "ask_user": "ask_user",
            "create_research_jobs": "create_research_jobs",
            "ready_for_dispatch": "ready_for_dispatch",
            "needs_input": "needs_input",
            "__end__": END,
        },
    )
    workflow.add_edge("ask_user", "requirements_check")
    workflow.add_conditional_edges(
        "ready_for_dispatch",
        dispatch_research,
        ["research_worker"],
    )
    workflow.add_edge("needs_input", END)
    workflow.add_edge("research_worker", "check_research_completeness")
    workflow.add_conditional_edges(
        "check_research_completeness",
        route_research_completeness,
        {
            "prepare_itinerary_context": "prepare_itinerary_context",
            "__end__": END,
        },
    )
    workflow.add_edge("prepare_itinerary_context", "final_planner")
    workflow.add_edge("final_planner", "validate_final")
    workflow.add_conditional_edges(
        "validate_final",
        route_review,
        {"__end__": END, "review_again": "final_planner"},
    )
    return workflow.compile(checkpointer=InMemorySaver())


def read_trip_request() -> str:
    raw_request = input("Enter your trip request (blank for Kyoto demo): ")
    if not raw_request.strip():
        print("Using the Kyoto demo request.")
        return DEMO_REQUEST
    return raw_request


async def main() -> None:
    from langchain.mcp import MCPAdapter

    from .accommodation_agent import build_accommodation_graph
    from .activities_graph import build_activities_graph

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.getenv("WANDER_DEBUG", "").strip().casefold() in {"1", "true", "yes"}:
        logging.getLogger("wander.graph").setLevel(logging.DEBUG)
    model = ChatOllama(model=MODEL_NAME, temperature=0, reasoning=False)
    intent_model = model.with_structured_output(TripIntent, method="json_schema")

    callbacks: list[Any] = []
    langfuse_client = None
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        langfuse_client = get_client()
        callbacks.append(CallbackHandler())
    except ImportError:
        pass

    project_root = Path(__file__).resolve().parents[2]
    server = server_parameters(project_root)
    async with MCPAdapter({"mcpServers": {"wander": {
        "transport": "stdio", "command": server.command, "args": server.args,
        "cwd": server.cwd, "env": server.env,
    }}}) as adapter:
        discovered = {tool.name: tool for tool in await adapter.list_tools()}
        needed = {
            name: discovered[name]
            for name in ("search_attractions", "search_hotels")
            if name in discovered
        }
        if len(needed) != 2:
            raise RuntimeError(
                "The MCP server must expose search_attractions and search_hotels."
            )
        schemas = {name: tool_schema(tool) for name, tool in needed.items()}
        activities_graph = await build_activities_graph(
            [needed["search_attractions"]], model
        )
        accommodation_graph = await build_accommodation_graph(
            [needed["search_hotels"]], model
        )
        node_context = TripNodeContext(
            model=model,
            intent_model=intent_model,
            schemas=schemas,
            live_providers=travel_settings().provider == "live",
            activities_graph=activities_graph,
            accommodation_graph=accommodation_graph,
        )

        graph = build_workflow(node_context)
        config: dict[str, Any] = {
            "configurable": {"thread_id": str(uuid4())},
            "recursion_limit": 50,
        }
        if callbacks:
            config["callbacks"] = callbacks

        request = read_trip_request()
        print(f"Trip request: {request!r}", flush=True)
        logger.debug("Trip request received: %r", request)
        logger.info("Starting trip workflow")
        result = await graph.ainvoke(
            {
                "user_request": request,
                "issues": [],
                "job_answers": {},
                "plan_attempts": 0,
                "clarification_rounds": 0,
                "status": "pending",
            },
            config=config,
        )

        while result.get("__interrupt__"):
            logger.info("Trip workflow paused for user clarification")
            payload = result["__interrupt__"][0].value
            answers: dict[str, dict[str, str]] = {}
            print("\nMissing or invalid trip details:")
            for item in payload["fields"]:
                job_id = item["job_id"]
                field = item["field"]
                print(f"\n{item['destination']} - {job_id}" if item.get("destination") else "\nTrip requirements")
                print(f"{field}: {item['description']}")
                if choices := item.get("date_choice"):
                    print(f"  Or enter job_id.check_in=YYYY-MM-DD or job_id.check_out=YYYY-MM-DD ({choices}).")
                try:
                    value = input("  Your answer: ").strip()
                except EOFError:
                    print("\nTrip planning paused. The listed details are still required.")
                    return
                answers.setdefault(job_id, {})[field] = value
            logger.info("Resuming trip workflow after user clarification")
            result = await graph.ainvoke(Command(resume=answers), config=config)

        logger.info(
            "Trip workflow finished with status=%s",
            result.get("status", "unknown"),
        )
        print("\nStatus:", result.get("status", "unknown"))
        if result.get("status_message"):
            print(result["status_message"])
        if result.get("status") == "needs_input":
            print(
                "Could not complete required details:",
                json.dumps(result.get("missing_details", []), indent=2),
            )
        if result.get("research_coverage"):
            coverage = result["research_coverage"]
            print(f"Research: {coverage['completed']} usable / {coverage['expected']} expected")
        if result.get("plan_attempts"):
            print("Planning attempts:", result.get("plan_attempts", 0))
            print("Validation issues:", json.dumps(result.get("issues", []), indent=2))
        if result.get("research_warnings"):
            print("Research limitations:")
            for warning in result["research_warnings"]:
                print(f"- {warning}")
        if result.get("status") == "invalid":
            print("The itinerary draft failed validation.")
        elif result.get("structured_plan") is not None:
            print(
                "\nFinal itinerary:\n",
                json.dumps(result["structured_plan"], indent=2, ensure_ascii=False),
            )
        elif result.get("status") in {"research_complete", "partial"}:
            for record in result.get("research_results", []):
                if record.get("status") == "completed":
                    print(f"\n{record['destination']} - {record['agent']}")
                    print(json.dumps(record["result"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
