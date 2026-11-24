from copy import deepcopy
from typing import Any

from langgraph.types import interrupt

from ..schemas import TripState
from ..validation import normalize_value
from .context import TripNodeContext
from .job_creation import create_research_jobs_node


def create_ask_user_node(context: TripNodeContext):
    create_research_jobs = create_research_jobs_node(context)

    async def ask_user(state: TripState) -> dict[str, Any]:
        answers = interrupt(
            {
                "type": "missing_job_details",
                "fields": state["missing_details"],
                "message": "Please provide the requested details so research can continue.",
            }
        )
        if not isinstance(answers, dict):
            raise ValueError("Expected answers grouped by job_id.")

        job_answers = {
            str(job_id): dict(fields)
            for job_id, fields in state.get("job_answers", {}).items()
            if isinstance(fields, dict)
        }

        jobs_by_id = {
            str(job.get("job_id")): job
            for job in state.get("research_jobs", [])
            if isinstance(job, dict) and job.get("job_id")
        }
        updated_trip_intent = None
        for job_id, fields in answers.items():
            if str(job_id) == "__trip__":
                if not isinstance(fields, dict):
                    continue
                current_intent = state.get("trip_intent", {})
                if isinstance(current_intent, dict):
                    updated_trip_intent = deepcopy(current_intent)
                else:
                    dump = getattr(current_intent, "model_dump", None)
                    updated_trip_intent = dump() if callable(dump) else {}
                for field, raw_value in fields.items():
                    if field == "destination":
                        try:
                            destination = normalize_value(
                                raw_value, {"type": "string"}, field
                            )
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(destination, str) or not destination.strip():
                            continue
                        existing = updated_trip_intent.get("destinations")
                        if not isinstance(existing, list) or not any(
                            isinstance(item, str) and item.strip()
                            for item in existing
                        ):
                            updated_trip_intent["destinations"] = [destination]
                    elif field == "trip_days":
                        try:
                            updated_trip_intent["trip_days"] = normalize_value(
                                raw_value, {"type": "integer"}, field
                            )
                        except (ValueError, TypeError):
                            updated_trip_intent["trip_days"] = raw_value
                    elif field == "research_scope":
                        scope = str(raw_value).strip().casefold()
                        if scope in {"activities", "activity", "sightseeing", "attractions"}:
                            updated_trip_intent["needs_activities"] = True
                            updated_trip_intent["needs_accommodation"] = False
                        elif scope in {"accommodation", "hotel", "hotels", "lodging"}:
                            updated_trip_intent["needs_activities"] = False
                            updated_trip_intent["needs_accommodation"] = True
                        elif scope in {"both", "all"}:
                            updated_trip_intent["needs_activities"] = True
                            updated_trip_intent["needs_accommodation"] = True
                continue

            job = jobs_by_id.get(str(job_id))
            if not isinstance(fields, dict) or job is None:
                continue
            tool_schema = context.schemas.get(str(job.get("tool_name")))
            if not tool_schema:
                continue
            properties = tool_schema.get("properties", {})
            target = job_answers.setdefault(str(job_id), {})
            for field, raw_value in fields.items():
                if field not in properties:
                    continue
                try:
                    target[field] = normalize_value(
                        raw_value,
                        properties[field],
                        field,
                    )
                except (ValueError, TypeError):
                    target[field] = raw_value

        update = {
            "job_answers": job_answers,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }
        if updated_trip_intent is not None:
            update["trip_intent"] = updated_trip_intent
        if not state.get("research_jobs"):
            update.update(await create_research_jobs({
                **state,
                "trip_intent": (
                    updated_trip_intent
                    if updated_trip_intent is not None
                    else state.get("trip_intent", {})
                ),
            }))
        return update

    return ask_user
