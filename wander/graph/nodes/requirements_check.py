import logging
from copy import deepcopy
from datetime import date
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from ..schemas import TripState
from ..validation import missing, normalize_value, validated_destinations
from .context import TripNodeContext


logger = logging.getLogger(__name__)
GLOBAL_JOB_ID = "__trip__"


def trip_issue(field: str, description: str) -> dict[str, str]:
    return {"job_id": GLOBAL_JOB_ID, "destination": "", "tool": "",
            "field": field, "description": description}


def _global_issues(intent: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    try:
        destinations = validated_destinations(intent.get("destinations"))
    except ValueError:
        destinations = []
        issues.append(trip_issue(
            "destinations",
            "Which destination(s) would you like to visit? Separate stops with 'and' or a semicolon.",
        ))
    if type(intent.get("requires_itinerary")) is not bool:
        issues.append(trip_issue(
            "requires_itinerary", "Would you like a day-by-day itinerary or search results? Enter itinerary or search.",
        ))
    elif intent["requires_itinerary"]:
        days = intent.get("trip_days")
        places = " across " + " and ".join(destinations) if destinations else ""
        if type(days) is not int or days <= 0:
            issues.append(trip_issue("trip_days", f"How many total days would you like to spend{places}?"))
        elif days < len(destinations):
            issues.append(trip_issue(
                "trip_days", f"Enter at least {len(destinations)} itinerary days for {', '.join(destinations)}.",
            ))
    return issues


def _hotel_schedule(jobs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], date, date]]:
    return [
        (job, date.fromisoformat(job["arguments"]["check_in"]),
         date.fromisoformat(job["arguments"]["check_out"]))
        for job in jobs
        if job.get("tool_name") == "search_hotels"
        and job["arguments"].get("check_in") and job["arguments"].get("check_out")
    ]


def hotel_day_minimums(jobs: list[dict[str, Any]]) -> dict[str, int]:
    """Map the time between ordered check-ins to minimum itinerary allocations."""
    schedule = _hotel_schedule(jobs)
    return {
        job["destination"]: ((schedule[index + 1][1] if index + 1 < len(schedule) else end) - start).days
        for index, (job, start, end) in enumerate(schedule)
    }


def _schedule_issues(jobs: list[dict[str, Any]], intent: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    schedule = _hotel_schedule(jobs)
    for (previous, _, checkout), (job, checkin, _) in zip(schedule, schedule[1:]):
        if checkin < checkout:
            issues.append({
                "job_id": job["job_id"], "destination": job["destination"], "tool": "search_hotels",
                "field": "check_in",
                "description": f"Check-in must be on or after {previous['destination']} check-out ({checkout}).",
            })
    if not issues and schedule and intent.get("requires_itinerary"):
        start, end = schedule[0][1], schedule[-1][2]
        nights = (end - start).days
        if nights > intent["trip_days"]:
            jobs_with_dates = {job["job_id"]: job for job, _, _ in schedule}
            issues.append({
                **trip_issue(
                    "trip_days",
                    f"The lodging schedule spans {nights} nights ({start} to {end}). "
                    "Enter a total itinerary duration that can cover it, or choose a lodging date to correct.",
                ),
                "date_choice": ", ".join(jobs_with_dates),
            })
    return issues


def create_requirements_check_node(context: TripNodeContext):
    def requirements_check(state: TripState) -> dict[str, Any]:
        intent = state.get("trip_intent")
        if isinstance(intent, dict):
            if not (intent.get("needs_activities") or intent.get("needs_accommodation")):
                if intent.get("requires_itinerary") is False:
                    message = (
                        "This request asks for neither attraction nor accommodation research."
                        if intent.get("destinations")
                        else "WanderMind supports trip planning, attraction searches, and accommodation searches."
                    )
                    return {"missing_details": [], "status": "no_research", "status_message": message}
                return {"validated_jobs": [], "missing_details": [trip_issue(
                    "research_scope", "What should we research? Enter attractions, hotels, or both.",
                )]}
            global_issues = _global_issues(intent)
            if global_issues:
                logger.info("Global requirements need clarification: %s",
                            [item["field"] for item in global_issues])
                return {"validated_jobs": [], "missing_details": global_issues}
            if "research_jobs" in state and not state["research_jobs"]:
                return {"missing_details": [], "status": "workflow_error",
                        "status_message": "No research jobs were generated for the validated trip intent."}

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
            schema = (context.schemas or {}).get(tool_name)

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
                    if field == "destination" and destination and converted != destination:
                        raise ValueError(f"Use the destination assigned to this job, {destination}.")
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
                    description = properties.get(field, {}).get("description") or {
                        "check_in": "What is the hotel check-in date (YYYY-MM-DD)?",
                        "check_out": "What is the hotel check-out date (YYYY-MM-DD)?",
                        "budget": "What is your nightly hotel budget in USD?",
                    }.get(field, f"Please provide {field}.")
                    issues.append({
                        "job_id": job_id,
                        "destination": destination,
                        "tool": tool_name,
                        "field": field,
                        "description": str(description),
                    })

            start = valid.get("check_in")
            end = valid.get("check_out")
            if tool_name == "search_hotels" and start and end and end <= start:
                issues.append({
                    "job_id": job_id,
                    "destination": destination,
                    "tool": tool_name,
                    "field": "check_out",
                    "description": "Check-out must be later than check-in.",
                })

            validated_job["arguments"] = valid
            validated_jobs.append(validated_job)

        if not issues:
            issues.extend(_schedule_issues(validated_jobs, intent or {}))
        unique: dict[tuple[str, str], dict[str, str]] = {}
        for item in issues:
            unique[(item["job_id"], item["field"])] = item
        logger.info("Research requirements: jobs=%d pending_fields=%d", len(validated_jobs), len(unique))
        result = {
            "validated_jobs": validated_jobs,
            "missing_details": list(unique.values()),
        }
        if any(item["field"].startswith("_") for item in issues):
            result.update(status="workflow_error", status_message="Research job configuration is invalid.")
        return result

    return requirements_check
