import re
from typing import Any

from langgraph.types import interrupt

from ..schemas import TripState
from ..validation import missing, parse_date, parse_number, validated_destinations
from .context import TripNodeContext
from .requirements_check import GLOBAL_JOB_ID


def create_ask_user_node(_context: TripNodeContext):
    def ask_user(state: TripState) -> dict[str, Any]:
        fields = state["missing_details"]
        answers = interrupt({
            "type": "missing_job_details",
            "fields": fields,
            "message": "Please provide the requested details so research can continue.",
        })
        if not isinstance(answers, dict):
            answers = {}

        intent = dict(state.get("trip_intent", {}))
        job_answers = {
            job_id: dict(values)
            for job_id, values in state.get("job_answers", {}).items()
            if isinstance(values, dict)
        }
        for item in fields:
            job_id, field = item["job_id"], item["field"]
            provided = answers.get(job_id)
            value = provided.get(field) if isinstance(provided, dict) else None
            if missing(value):
                continue
            if job_id != GLOBAL_JOB_ID:
                # The requirements node is the single authority for normalization.
                job_answers.setdefault(job_id, {})[field] = value
                continue

            try:
                if field == "destinations":
                    stops = re.split(r"\s+\band\b\s+|;", value, flags=re.I) if isinstance(value, str) else value
                    intent[field] = [stop.strip() for stop in validated_destinations(stops)]
                elif field == "trip_days":
                    date_correction = re.fullmatch(
                        r"(accommodation-\d+)\.(check_in|check_out)\s*=\s*(.+)", value.strip()
                    ) if isinstance(value, str) and item.get("date_choice") else None
                    if date_correction and date_correction[1] in item["date_choice"].split(", "):
                        date_job, date_field, date_value = date_correction.groups()
                        job_answers.setdefault(date_job, {})[date_field] = parse_date(date_value)
                    else:
                        intent[field] = parse_number(value, integer=True, field=field)
                elif field == "research_scope" and isinstance(value, str):
                    scope = value.strip().casefold()
                    if scope in {"attractions", "hotels", "both"}:
                        intent.update(
                            needs_activities=scope in {"attractions", "both"},
                            needs_accommodation=scope in {"hotels", "both"},
                        )
                        if intent.get("requires_itinerary") is None:
                            intent["requires_itinerary"] = False
                elif field == "requires_itinerary" and isinstance(value, str):
                    choice = value.strip().casefold()
                    if choice in {"itinerary", "search"}:
                        intent[field] = choice == "itinerary"
            except ValueError:
                # Keep the previous value and let deterministic validation ask again.
                continue

        return {
            "trip_intent": intent,
            "job_answers": job_answers,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }

    return ask_user
