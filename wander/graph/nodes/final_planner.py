import json
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ..schemas import TripItinerary, TripState
from .context import TripNodeContext
from .validate_final import _entity_name, _hotel_over_budget, _result_content, _structured_records


_PROMPT = """Build a structured itinerary using only the supplied research summaries.
Fill the provided day slots in their existing order. Copy each slot's exact
day_number and destination to the result. The original request supplies
preferences, and the validated context determines the schedule. Spread distinct, researched attractions across
sightseeing days before adding extra stops to any day. Include an activity on
each allocated day when enough distinct supported places exist in that city.
Each suggested_activity shows an evidence-backed place for that day. Follow
these placements unless making a specific improvement that preserves distinct
sightseeing coverage on every suggested day.
If there are fewer places, distribute the available ones without repeating or
inventing places, leave unsupported slots open, and explain the shortage.
Use each supplied attraction name exactly and put its supporting job ID in
evidence_job_id. Add transfer notes only when useful. Do not invent schedules,
journey durations, hours, admission prices, or visit durations. Transfer notes
and source warnings are added by the workflow.

For lodging, use a researched candidate if one is suitable. The accommodation
field is the exact property name only. Put the supporting ID in
accommodation_evidence_job_id. An unknown
price means budget fit is unknown. Never imply that rates or room availability
were checked when the evidence does not say so. Leave both accommodation fields
null on slots with lodging_night=false, including the checkout day. Use an
evidence-backed candidate when lodging_night=true and one is available. Choose
one property for each destination's stay and keep that choice for its nights. Mock
prices and places are illustrative. Never invent details to fill a gap.
"""


def _planning_schema(context: dict[str, Any], state: TripState) -> dict[str, Any]:
    """Constrain the existing itinerary format to names in specialist reports."""
    schema = TripItinerary.model_json_schema()
    day = schema["$defs"]["ItineraryDay"]["properties"]
    activity = schema["$defs"]["PlannedActivity"]["properties"]
    day["destination"]["enum"] = context["destinations"]
    schema["properties"]["days"].update(minItems=context["total_days"], maxItems=context["total_days"])
    schema["properties"]["warnings"]["maxItems"] = 0
    activity["notes"] = day["transfer_notes"] = {"type": "null", "default": None}

    groups = context["research_by_destination"]
    lodging = {stay["job_id"]: stay for stay in context["lodging_constraints"]}
    for agent in ("activities", "accommodation"):
        names: dict[str, None] = {}
        jobs: dict[str, None] = {}
        enough_for_each_day = True
        for destination, evidence in groups.items():
            destination_choices: dict[str, str] = {}
            for report in evidence[agent]:
                records, _ = _structured_records(_result_content(report))
                for record in records:
                    job_id = report["job_id"]
                    if agent == "accommodation" and _hotel_over_budget(
                        state, record, {"arguments": lodging.get(job_id, {})},
                    ):
                        continue
                    if name := _entity_name(record).strip():
                        names[name] = None
                        destination_choices.setdefault(name, job_id)
                        jobs[job_id] = None
                        if agent == "accommodation":
                            # Use the highest-ranked suitable candidate for this whole stay.
                            break
                if agent == "accommodation" and destination_choices:
                    break
            enough_for_each_day &= len(destination_choices) >= context["day_allocation"][destination]
            if agent == "activities":
                slots = (slot for slot in context["day_slots"] if slot["destination"] == destination)
                for slot, (name, job_id) in zip(slots, destination_choices.items()):
                    slot["suggested_activity"] = {"name": name, "evidence_job_id": job_id}
        if agent == "activities":
            if names:
                activity["name"]["enum"] = list(names)
                activity["evidence_job_id"]["enum"] = list(jobs)
                if enough_for_each_day:
                    day["activities"]["minItems"] = 1
                    schema["$defs"]["ItineraryDay"]["required"].append("activities")
            else:
                day["activities"]["maxItems"] = 0
        else:
            for field, values in (("accommodation", names), ("accommodation_evidence_job_id", jobs)):
                branches = [{"type": "string", "enum": list(values)}] if values else []
                day[field]["anyOf"] = [*branches, {"type": "null"}]
    return schema


def create_final_planner_node(context: TripNodeContext):
    async def final_planner(state: TripState) -> dict[str, Any]:
        context_data = state["itinerary_context"]
        planning_context = {**context_data, "day_slots": [dict(slot) for slot in context_data["day_slots"]]}
        structured_model = context.model.with_structured_output(
            _planning_schema(planning_context, state), method="json_schema",
        )
        messages = [
            SystemMessage(content=_PROMPT),
            HumanMessage(content=(
                f"Original request:\n{state['user_request']}\n\n"
                f"Validated itinerary context and research:\n{json.dumps(planning_context, ensure_ascii=False)}"
            )),
        ]
        if state.get("issues"):
            messages.append(HumanMessage(content=(
                f"Previous plan:\n{json.dumps(state.get('structured_plan'), ensure_ascii=False)}\n\n"
                f"Correct these specific planning defects:\n{json.dumps(state['issues'])}"
            )))

        planner_error = None
        try:
            response = await structured_model.ainvoke(messages)
            plan = TripItinerary.model_validate(response).model_dump(mode="json")
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False)
            ]
            planner_error = "The planner response violated the itinerary schema. " + ". ".join(problems)
            plan = None
        except OutputParserException:
            planner_error = "The planner response could not be parsed as a structured itinerary."
            plan = None

        return {
            "structured_plan": plan,
            "planner_error": planner_error,
            "plan_attempts": state.get("plan_attempts", 0) + 1,
        }

    return final_planner
