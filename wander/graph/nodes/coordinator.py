import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import ResearchTasks, TripState
from ..validation import clean_task, explicit_nightly_budget
from .context import TripNodeContext


logger = logging.getLogger(__name__)


def create_coordinator_node(context: TripNodeContext):
    async def coordinator(state: TripState) -> dict[str, Any]:
        explicit_context = state.get("user_context", {})
        response = await context.coordinator_model.ainvoke(
            [
                SystemMessage(
                    content="""
                    You are WanderMind's Trip Coordinator.

                    Your responsibility is to decide which specialist
                    agents are needed to fulfill the user's request.

                    Available agents:

                    1. activities:
                    Researches attractions, sightseeing, experiences,
                    and activities.

                    2. accommodation:
                    Researches hotels and places to stay.

                    TASK SELECTION RULES:

                    - If the user requests a complete trip, vacation,
                    or travel itinerary, select BOTH agents.

                    - If the user only asks about attractions or
                    experiences, select ONLY activities.

                    - If the user only asks about hotels or
                    accommodation, select ONLY accommodation.

                    - Set unused tasks to null.

                    IMPORTANT:
                    - Missing tool arguments do NOT mean an agent
                    should be skipped.
                    - For example, if hotel dates are missing,
                    still create the accommodation task.
                    - The Requirements Check node will request
                    missing details from the user.

                    - Never invent dates, budgets, or destinations.
                    - Preserve explicit user corrections.
                    - Only return both tasks as null when the user
                    request is unrelated to available specialists.

                    EXAMPLES:

                    User: "Plan a 3-day Kyoto trip"
                    activities_task: "Research attractions in Kyoto"
                    accommodation_task: "Research hotels in Kyoto"

                    User: "Find temples in Kyoto"
                    activities_task: "Research temples in Kyoto"
                    accommodation_task: null

                    User: "Find hotels in Osaka"
                    activities_task: null
                    accommodation_task: "Research hotels in Osaka"
                    """
                ),
                HumanMessage(
                    content=(
                        f"Original request:\n{state['user_request']}\n\n"
                        "Latest validated user clarifications:\n"
                        f"{json.dumps(explicit_context)}\n\n"
                        f"Trip Intent: {state['trip_intent']}"
                        f"Available tools:\n{json.dumps(context.tool_specs)}"
                    )
                ),
            ]
        )

        logger.info(
            "Coordinator structured output:\n%s",
            response.model_dump_json(indent=2),
        )
        if not isinstance(response, ResearchTasks):
            raise RuntimeError("Coordinator did not return validated ResearchTasks.")

        intent = state["trip_intent"]
        destinations = intent.get("destinations", [])
        args = {
            name: dict(response.tool_arguments.get(name) or {})
            for name in context.schemas
        }

        if len(destinations) == 1:
            destination = destinations[0]
            for tool_name, schema in context.schemas.items():
                if "destination" in schema.get("properties", {}):
                    args[tool_name]["destination"] = destination

        budget = intent.get("nightly_hotel_budget")
        if budget is not None and "search_hotels" in args:
            args["search_hotels"]["budget"] = budget

        for tool_name, fields in explicit_context.items():
            if tool_name in args:
                args[tool_name].update(fields)

        nightly = explicit_nightly_budget(state["user_request"])
        if nightly is not None:
            args["search_hotels"]["budget"] = nightly
        elif re.search(
            r"\b(?:total|overall|entire)\s+(?:trip\s+)?budget\b",
            state["user_request"],
            re.IGNORECASE,
        ):
            args["search_hotels"].pop("budget", None)

        for name, fields in explicit_context.items():
            if name in args:
                args[name].update(fields)

        return {
            "activities_task": clean_task(response.activities_task),
            "accommodation_task": clean_task(response.accommodation_task),
            "tool_arguments": args,
        }

    return coordinator
