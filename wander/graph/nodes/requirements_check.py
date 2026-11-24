from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from ..schemas import TripState
from ..validation import missing, normalize_value
from .context import TripNodeContext


def create_requirements_check_node(context: TripNodeContext):
    def requirements_check(state: TripState) -> dict[str, Any]:
        validated_jobs: list[dict[str, Any]] = []
        issues: list[dict[str, str]] = []
        job_answers = state.get("job_answers", {})

        for job in state.get("research_jobs", []):
            if not isinstance(job, dict):
                issues.append({
                    "job_id": "unknown",
                    "destination": "",
                    "tool": "",
                    "field": "_job",
                    "description": "Malformed research job; expected a job object.",
                })
                continue

            job_id = str(job.get("job_id") or "unknown")
            destination = str(job.get("destination") or "")
            tool_name = str(job.get("tool_name") or "")
            validated_job = deepcopy(job)
            schema = context.schemas.get(tool_name)

            if not schema:
                issues.append({
                    "job_id": job_id,
                    "destination": destination,
                    "tool": tool_name,
                    "field": "_tool",
                    "description": f"Unknown or unsupported MCP tool: {tool_name or '(missing)'}.",
                })
                validated_job["arguments"] = deepcopy(job.get("arguments", {}))
                validated_jobs.append(validated_job)
                continue

            original = job.get("arguments", {})
            if not isinstance(original, dict):
                original = {}
                issues.append({
                    "job_id": job_id,
                    "destination": destination,
                    "tool": tool_name,
                    "field": "_arguments",
                    "description": "Job arguments must be an object.",
                })
            corrections = job_answers.get(job_id, {})
            if not isinstance(corrections, dict):
                corrections = {}
            merged = deepcopy(original)
            merged.update(deepcopy(corrections))

            properties = schema.get("properties", {})
            valid: dict[str, Any] = {}
            invalid_fields: set[str] = set()
            for field, value in merged.items():
                if field not in properties:
                    issues.append({
                        "job_id": job_id,
                        "destination": destination,
                        "tool": tool_name,
                        "field": field,
                        "description": f"Unsupported field for {tool_name}: {field}.",
                    })
                    continue
                if missing(value):
                    continue
                try:
                    converted = normalize_value(value, properties[field], field)
                    Draft202012Validator(
                        {
                            "type": "object",
                            "properties": {field: properties[field]},
                            "$defs": schema.get("$defs", {}),
                        },
                        format_checker=FormatChecker(),
                    ).validate({field: converted})
                    valid[field] = converted
                except (ValueError, TypeError, ValidationError) as exc:
                    detail = exc.message if isinstance(exc, ValidationError) else str(exc)
                    invalid_fields.add(field)
                    issues.append({
                        "job_id": job_id,
                        "destination": destination,
                        "tool": tool_name,
                        "field": field,
                        "description": f"Invalid {field}: {detail}",
                    })

            for field in schema.get("required", []):
                if field not in valid and field not in invalid_fields:
                    description = (
                        properties.get(field, {}).get("description")
                        or f"Please provide {field}."
                    )
                    issues.append({
                        "job_id": job_id,
                        "destination": destination,
                        "tool": tool_name,
                        "field": field,
                        "description": str(description),
                    })

            start = valid.get("check_in")
            end = valid.get("check_out")
            if tool_name == "search_hotels" and start and end:
                try:
                    from datetime import date
                    nights = (date.fromisoformat(end) - date.fromisoformat(start)).days
                    if nights <= 0:
                        issues.append({
                            "job_id": job_id,
                            "destination": destination,
                            "tool": tool_name,
                            "field": "check_out",
                            "description": "Check-out must be later than check-in.",
                        })
                except (TypeError, ValueError):
                    pass

            validated_job["arguments"] = valid
            validated_jobs.append(validated_job)

        trip_intent = state.get("trip_intent", {})
        if not isinstance(trip_intent, dict):
            dump = getattr(trip_intent, "model_dump", None)
            trip_intent = dump() if callable(dump) else {}
        destinations = trip_intent.get("destinations")
        needs_activities = trip_intent.get("needs_activities") is True
        needs_accommodation = trip_intent.get("needs_accommodation") is True
        if (needs_activities or needs_accommodation) and not (
            isinstance(destinations, list)
            and any(isinstance(item, str) and item.strip() for item in destinations)
        ):
            issues.append({
                "job_id": "__trip__",
                "destination": "",
                "tool": "trip_intent",
                "field": "destination",
                "description": "Which destination would you like to plan?",
            })

        trip_days = trip_intent.get("trip_days")
        if needs_activities and (
            not isinstance(trip_days, int)
            or isinstance(trip_days, bool)
            or trip_days <= 0
        ):
            issues.append({
                "job_id": "__trip__",
                "destination": "",
                "tool": "trip_intent",
                "field": "trip_days",
                "description": "How many total days would you like to plan?",
            })

        if not state.get("research_jobs") and not issues:
            issues.append({
                "job_id": "__trip__",
                "destination": "",
                "tool": "trip_intent",
                "field": "research_scope",
                "description": (
                    "Would you like activities research, accommodation research, or both?"
                ),
            })

        unique: dict[tuple[str, str], dict[str, str]] = {}
        for item in issues:
            unique[(item["job_id"], item["field"])] = item
        return {
            "validated_jobs": validated_jobs,
            "missing_details": list(unique.values()),
        }

    return requirements_check
