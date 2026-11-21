from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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
    ResearchTasks,
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
logger = logging.getLogger(__name__)

__all__ = [
    "MAX_CLARIFICATION_ROUNDS",
    "MAX_PLAN_ATTEMPTS",
    "MODEL_NAME",
    "ResearchTasks",
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
    workflow.add_edge("extract_trip_intent", "create_research_jobs")
    workflow.add_edge("create_research_jobs", "coordinator")
    workflow.add_edge("coordinator", "requirements_check")
    workflow.add_conditional_edges(
        "requirements_check",
        route_requirements,
        {
            "ask_user": "ask_user",
            "ready_for_dispatch": "ready_for_dispatch",
            "needs_input": "needs_input",
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
            "needs_input": "needs_input",
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


async def main() -> None:
    from langchain.mcp import MCPAdapter

    from .accommodation_agent import build_accommodation_graph
    from .activities_graph import build_activities_graph

    load_dotenv()
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    coordinator_model = model.with_structured_output(ResearchTasks)
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

    server_path = Path(__file__).resolve().parent.parent / "mcp_server.py"
    async with MCPAdapter(server_path) as adapter:
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
        tool_specs = [
            {
                "name": name,
                "description": tool.description,
                "schema": schemas[name],
            }
            for name, tool in needed.items()
        ]

        activities_graph = await build_activities_graph(
            [needed["search_attractions"]]
        )
        accommodation_graph = await build_accommodation_graph(
            [needed["search_hotels"]]
        )
        node_context = TripNodeContext(
            model=model,
            coordinator_model=coordinator_model,
            validator_model=None,
            intent_model=intent_model,
            schemas=schemas,
            tool_specs=tool_specs,
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

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        request = input("Enter your trip request (blank for Kyoto demo): ").strip()
        if not request:
            request = "Plan a 3-day Kyoto trip with a nightly hotel budget of $150."
        logger.info("Starting trip workflow")
        logger.info("Workflow route: START -> extract_trip_intent")
        result = await graph.ainvoke(
            {
                "user_request": request,
                "issues": [],
                "input_errors": [],
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
            print("\nMissing or invalid research details:")
            for item in payload["fields"]:
                job_id = item["job_id"]
                destination = item.get("destination") or "Unknown destination"
                field = item["field"]
                print(f"\n{destination} - {job_id}")
                print(f"{field}: {item['description']}")
                value = input("  Your answer: ").strip()
                answers.setdefault(job_id, {})[field] = value
            logger.info("Resuming trip workflow after user clarification")
            result = await graph.ainvoke(Command(resume=answers), config=config)

        logger.info(
            "Trip workflow finished with status=%s",
            result.get("status", "unknown"),
        )
        print("\nStatus:", result.get("status", "unknown"))
        if result.get("status") == "needs_input":
            print(
                "Could not complete required details:",
                json.dumps(result.get("missing_details", []), indent=2),
            )
        else:
            print("Planning attempts:", result.get("plan_attempts", 0))
            print("Validation issues:", json.dumps(result.get("issues", []), indent=2))
            print(
                "\nFinal itinerary:\n",
                json.dumps(result.get("structured_plan"), indent=2),
            )

if __name__ == "__main__":
    asyncio.run(main())
