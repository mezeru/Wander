from typing import Any

from ..schemas import TripIntent, TripState


_RESEARCH_CATEGORIES = ("activities", "accommodation")


def _validated_destinations(destinations: Any) -> list[str]:
    if not isinstance(destinations, list) or not destinations:
        raise ValueError("At least one destination is required.")

    normalized: set[str] = set()
    for destination in destinations:
        if not isinstance(destination, str) or not destination.strip():
            raise ValueError("Destinations must be non-empty strings.")
        key = destination.strip().casefold()
        if key in normalized:
            raise ValueError(
                "Duplicate destinations are not supported; destination names "
                "must be unique ignoring case and surrounding whitespace."
            )
        normalized.add(key)
    return destinations


def _trip_intent_data(state: TripState) -> dict[str, Any]:
    trip_intent = state.get("trip_intent")
    if isinstance(trip_intent, TripIntent):
        return trip_intent.model_dump()
    if isinstance(trip_intent, dict):
        return trip_intent
    raise ValueError("Trip intent must be available to prepare itinerary context.")


def allocate_days(
    destinations: list[str],
    total_days: int,
) -> dict[str, int]:
    ordered_destinations = _validated_destinations(destinations)
    if isinstance(total_days, bool) or not isinstance(total_days, int) or total_days <= 0:
        raise ValueError("Trip length must be a positive integer number of days.")
    if total_days < len(ordered_destinations):
        raise ValueError(
            "Not enough days to allocate at least one day per destination."
        )

    base, remainder = divmod(total_days, len(ordered_destinations))
    return {
        destination: base + (index < remainder)
        for index, destination in enumerate(ordered_destinations)
    }


def _group_research(
    state: TripState,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    trip_intent = _trip_intent_data(state)
    destinations = _validated_destinations(trip_intent.get("destinations"))
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        destination: {category: [] for category in _RESEARCH_CATEGORIES}
        for destination in destinations
    }
    destination_set = set(destinations)
    assignment_counts: dict[str, int] = {}
    assignments: dict[str, dict[str, Any]] = {}
    valid_assignment_order: list[str] = []
    evidence_issues: list[dict[str, Any]] = []
    category_expected = {
        destination: {category: 0 for category in _RESEARCH_CATEGORIES}
        for destination in destinations
    }

    for job in state.get("validated_jobs", []):
        if not isinstance(job, dict):
            evidence_issues.append({"job_id": None, "reason": "malformed_assignment"})
            continue
        job_id = job.get("job_id")
        destination = job.get("destination")
        agent = job.get("agent")
        if not isinstance(destination, str) or destination not in destination_set:
            evidence_issues.append({"job_id": job_id, "reason": "unknown_assignment_destination"})
            continue
        if agent not in _RESEARCH_CATEGORIES:
            evidence_issues.append({"job_id": job_id, "reason": "unknown_assignment_agent"})
            continue
        category_expected[destination][agent] += 1
        if not isinstance(job_id, str) or not job_id:
            evidence_issues.append({"job_id": job_id, "reason": "invalid_assignment_job_id"})
            continue
        assignment_counts[job_id] = assignment_counts.get(job_id, 0) + 1
        if assignment_counts[job_id] > 1:
            evidence_issues.append({"job_id": job_id, "reason": "duplicate_assignment_job_id"})
            assignments.pop(job_id, None)
            continue
        assignments[job_id] = job
        valid_assignment_order.append(job_id)

    valid_assignment_order = [
        job_id for job_id in valid_assignment_order if job_id in assignments
    ]

    result_counts: dict[str, int] = {}
    results_by_job_id: dict[str, dict[str, Any]] = {}
    for result in state.get("research_results", []):
        if not isinstance(result, dict):
            evidence_issues.append({"job_id": None, "reason": "malformed_result"})
            continue
        job_id = result.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            evidence_issues.append({"job_id": job_id, "reason": "missing_result_job_id"})
            continue
        result_counts[job_id] = result_counts.get(job_id, 0) + 1
        if job_id not in assignments:
            reason = (
                "duplicate_assignment_job_id"
                if assignment_counts.get(job_id, 0) > 1
                else "unknown_result_job_id"
            )
            evidence_issues.append({"job_id": job_id, "reason": reason})
            continue
        if result_counts[job_id] > 1:
            evidence_issues.append({"job_id": job_id, "reason": "duplicate_result_job_id"})
            results_by_job_id.pop(job_id, None)
            continue
        results_by_job_id[job_id] = result

    duplicate_result_ids = {
        job_id for job_id, count in result_counts.items() if count > 1
    }
    usable_by_job_id: dict[str, dict[str, Any]] = {}
    for job_id in valid_assignment_order:
        if job_id in duplicate_result_ids:
            continue
        result = results_by_job_id.get(job_id)
        if result is None or result.get("status") != "completed":
            continue

        assignment = assignments[job_id]
        if (
            result.get("destination") != assignment["destination"]
            or result.get("agent") != assignment["agent"]
        ):
            evidence_issues.append({"job_id": job_id, "reason": "assignment_mismatch"})
            continue
        content = result.get("result")
        usable_content = (
            isinstance(content, str) and bool(content.strip())
            or isinstance(content, (dict, list)) and bool(content)
        )
        if not usable_content:
            evidence_issues.append({"job_id": job_id, "reason": "empty_completed_result"})
            continue
        usable_by_job_id[job_id] = result

    category_usable = {
        destination: {category: 0 for category in _RESEARCH_CATEGORIES}
        for destination in destinations
    }
    for job_id in valid_assignment_order:
        result = usable_by_job_id.get(job_id)
        if result is None:
            continue
        assignment = assignments[job_id]
        destination = assignment["destination"]
        agent = assignment["agent"]
        grouped[destination][agent].append(result)
        category_usable[destination][agent] += 1

    coverage = {
        destination: {
            category: {
                "expected": category_expected[destination][category],
                "usable": category_usable[destination][category],
                "status": (
                    "not_requested"
                    if category_expected[destination][category] == 0
                    else "complete"
                    if category_usable[destination][category]
                    == category_expected[destination][category]
                    else "incomplete"
                ),
            }
            for category in _RESEARCH_CATEGORIES
        }
        for destination in destinations
    }
    return grouped, coverage, sorted(
        evidence_issues,
        key=lambda item: (str(item.get("job_id")), item["reason"]),
    )


def group_research_by_destination(
    state: TripState,
) -> dict[str, Any]:
    grouped, _, _ = _group_research(state)
    return grouped


def prepare_itinerary_context(state: TripState) -> dict[str, Any]:
    trip_intent = _trip_intent_data(state)
    destinations = _validated_destinations(trip_intent.get("destinations"))
    total_days = trip_intent.get("trip_days")
    allocation = allocate_days(destinations, total_days)
    research_by_destination, destination_coverage, evidence_issues = _group_research(
        state
    )

    research_coverage = state.get("research_coverage", {})
    if not isinstance(research_coverage, dict):
        research_coverage = {}
    research_coverage = dict(research_coverage)
    research_coverage["by_destination"] = destination_coverage
    research_coverage["evidence_issues"] = evidence_issues
    return {
        "itinerary_context": {
            "destinations": destinations,
            "total_days": total_days,
            "day_allocation": allocation,
            "research_by_destination": research_by_destination,
            "research_coverage": research_coverage,
        }
    }
