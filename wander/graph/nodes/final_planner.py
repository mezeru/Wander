import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState, TripItinerary
from .context import TripNodeContext



def create_final_planner_node(context: TripNodeContext):
    with_structured_output = getattr(context.model, "with_structured_output", None)
    structured_model = (
        with_structured_output(
            TripItinerary,
            method="json_schema",
        )
        if callable(with_structured_output)
        else None
    )

    async def final_planner(state: TripState) -> dict[str, Any]:
        messages = [
            SystemMessage(
                content=(
                    "You are the WanderMind multi-destination itinerary planner. "
                    "Use only the supplied research evidence. "
                    "Follow the requested destination order and initial day allocation. "
                    "Produce exactly one itinerary entry per requested day. "
                    "Choose activities from the appropriate destination's research. "
                    "Consider all distinct attractions in the available research summaries and "
                    "distribute them across days where possible. Avoid unnecessary empty "
                    "sightseeing days when enough researched attractions exist. "
                    "Never invent attractions to fill the schedule; when research is sparse, "
                    "leave unsupported activities out and add a clear limitation to warnings. "
                    "Include intercity transfer notes when destinations change, "
                    "but do not invent transport schedules, prices, or journey durations. "
                    "Use evidence_job_id to reference supporting research. "
                    "Only recommend accommodation supported by accommodation research, and put "
                    "only the hotel name in accommodation (not evidence IDs or research notes). "
                    "Do not invent hotel names, prices, attractions, or availability. "
                    "Explicitly report missing research and uncertainty. "
                    "All mock research is illustrative, not live-verified."
                )
            ),
            HumanMessage(
                content=(
                    f"Original request:\n{state['user_request']}\n\n"
                    f"Validated arguments:\n{json.dumps(state.get('tool_arguments', {}))}\n\n"
                    f"Research results:\n{json.dumps(state.get('research_results', []), indent=2)}\n\n"
                    f"Research coverage and limitations:\n"
                    f"{json.dumps(state.get('research_coverage', {}), indent=2)}"
                )
            ),
        ]
        if state.get("issues"):
            messages.append(
                HumanMessage(
                    content=(
                        f"Previous plan:\n{json.dumps(state['structured_plan'], indent=2)}\n\n"
                        "Revise to address these validation issues:\n"
                        f"{json.dumps(state['issues'])}"
                    )
                )
            )
        coverage = state.get("research_coverage")
        if coverage:
            messages.append(
                HumanMessage(
                    content=(
                        f"Itinerary planning context:\n"
                        f"{json.dumps(state['itinerary_context'], indent=2)}"
                    )
                )
            )
        if structured_model is None:
            await context.model.ainvoke(messages)
            itinerary_context = state.get("itinerary_context", {})
            destinations = itinerary_context.get("destinations", [])
            allocation = itinerary_context.get("day_allocation", {})
            day_entries = []
            day_number = 1
            for destination in destinations:
                count = allocation.get(destination, 0)
                for _ in range(count):
                    day_entries.append({
                        "day_number": day_number,
                        "destination": destination,
                        "activities": [],
                        "accommodation": None,
                    })
                    day_number += 1
            response_data = {
                "title": "Itinerary",
                "days": day_entries,
                "warnings": [],
            }
        else:
            response = await structured_model.ainvoke(messages)
            response_data = (
                response.model_dump(mode="json")
                if isinstance(response, TripItinerary)
                else response
                if isinstance(response, dict)
                else {}
            )
        return {
            "structured_plan": response_data,
            "plan_attempts": state.get("plan_attempts", 0) + 1,
        }

    return final_planner
