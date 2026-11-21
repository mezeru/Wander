import json
import logging
import re
from typing import Any
from pydantic import ValidationError

from ..schemas import TripItinerary, TripState
from .check_research_completeness import execution_warnings, lodging_limitations
from .context import TripNodeContext
from .itinerary_context import group_research_by_destination


logger = logging.getLogger(__name__)


def _normalize_name(value: Any) -> str:
    return " ".join(value.casefold().split()) if isinstance(value, str) else ""


def validate_structure(state: TripState) -> tuple[TripItinerary | None, list[str]]:
    raw = state.get("structured_plan")
    if raw is None:
        return None, [state.get("planner_error") or "Structured itinerary is missing."]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, [f"Structured itinerary is invalid JSON: {exc.msg}."]
    if isinstance(raw, TripItinerary):
        plan = raw
    elif isinstance(raw, dict):
        try:
            plan = TripItinerary.model_validate(raw)
        except ValidationError as exc:
            return None, [
                "Structured itinerary is invalid: "
                + "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors()
                )
            ]
    else:
        return None, [
            "Structured itinerary must be a TripItinerary object, dictionary, or JSON object."
        ]
    return plan, []


def validate_day_constraints(
    plan: TripItinerary,
    context: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    total_days = context.get("total_days")
    if not isinstance(total_days, int) or isinstance(total_days, bool):
        return ["Itinerary context is missing a valid total_days value."]
    days = plan.days
    if len(days) != total_days:
        issues.append(
            f"Itinerary has {len(days)} days; exactly {total_days} days are required."
        )
    numbers = [day.day_number for day in days]
    expected = list(range(1, total_days + 1))
    if len(set(numbers)) != len(numbers):
        issues.append("Itinerary contains duplicate day numbers.")
    if numbers != expected:
        issues.append(
            "Day numbers must start at 1 and be consecutive by position "
            f"(expected {expected}, got {numbers})."
        )
    return issues


def validate_destination_allocation(
    plan: TripItinerary,
    context: dict[str, Any],
) -> list[str]:
    destinations = context.get("destinations")
    allocation = context.get("day_allocation")
    if not isinstance(destinations, list) or not isinstance(allocation, dict):
        return ["Itinerary context has invalid destinations or day_allocation data."]

    normalized_destinations = [
        _normalize_name(destination) for destination in destinations
        if isinstance(destination, str)
    ]
    expected_sequence: list[str] = []
    allocation_total = 0
    allocation_by_name: dict[str, int] = {}
    for destination in destinations:
        if not isinstance(destination, str):
            return ["Itinerary context contains a malformed destination."]
        count = allocation.get(destination)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return [f"Day allocation is invalid for destination '{destination}'."]
        allocation_by_name[_normalize_name(destination)] = count
        allocation_total += count
        expected_sequence.extend([_normalize_name(destination)] * count)

    total_days = context.get("total_days")
    if not isinstance(total_days, int) or isinstance(total_days, bool):
        return ["Itinerary context is missing a valid total_days value."]
    if allocation_total != total_days:
        issues = [
            f"Day allocation totals {allocation_total}, but the trip requires {total_days} days."
        ]
    else:
        issues = []

    actual_sequence = [_normalize_name(day.destination) for day in plan.days]
    if actual_sequence != expected_sequence:
        issues.append(
            "Destination order/allocation is incorrect "
            f"(expected {destinations_sequence_labels(destinations, allocation)}, "
            f"got {actual_sequence})."
        )

    extra = sorted(set(actual_sequence) - set(normalized_destinations))
    missing = sorted(set(normalized_destinations) - set(actual_sequence))
    if extra:
        issues.append("Itinerary contains extra destinations: " + ", ".join(extra))
    if missing:
        issues.append("Itinerary omits requested destinations: " + ", ".join(missing))
    return issues


def destinations_sequence_labels(
    destinations: list[Any],
    allocation: dict[str, Any],
) -> list[str]:
    return [
        str(destination)
        for destination in destinations
        for _ in range(allocation.get(destination, 0))
    ]


def _result_content(value: dict[str, Any]) -> Any:
    content = value.get("result")
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return content


def _structured_records(content: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(content, list):
        records = [item for item in content if isinstance(item, dict)]
        if records:
            return records, True
        if all(isinstance(item, str) for item in content):
            return [{"name": item} for item in content], True
        return [], False
    if not isinstance(content, dict):
        return [], False
    keys = ("attractions", "hotels", "accommodations", "activities", "results", "items", "data")
    for key in keys:
        nested = content.get(key)
        if isinstance(nested, list):
            records = [item for item in nested if isinstance(item, dict)]
            if records:
                return records, True
            if all(isinstance(item, str) for item in nested):
                return [{"name": item} for item in nested], True
    if any(_normalize_name(content.get(key)) for key in ("name", "hotel_name", "attraction_name", "title")):
        return [content], True
    return [], False


def _entity_name(record: dict[str, Any]) -> str:
    for key in ("name", "hotel_name", "attraction_name", "title", "entity_name"):
        if isinstance(record.get(key), str):
            return record[key]
    return ""


def validate_evidence_references(
    state: TripState,
    plan: TripItinerary,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    jobs: dict[str, dict[str, Any]] = {}
    duplicate_jobs: set[str] = set()
    for job in state.get("validated_jobs", []) or []:
        if not isinstance(job, dict):
            warnings.append("Research contains a malformed validated job.")
            continue
        job_id = job.get("job_id")
        if isinstance(job_id, str) and job_id:
            if job_id in jobs:
                duplicate_jobs.add(job_id)
            else:
                jobs[job_id] = job
    for job_id in duplicate_jobs:
        warnings.append(f"Duplicate authoritative research job ID '{job_id}'.")
        jobs.pop(job_id)

    counts: dict[str, int] = {}
    results: dict[str, dict[str, Any]] = {}
    for result in state.get("research_results", []) or []:
        if not isinstance(result, dict):
            warnings.append("Research contains a malformed result record.")
            continue
        job_id = result.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            warnings.append("Research result is missing a job_id.")
            continue
        counts[job_id] = counts.get(job_id, 0) + 1
        if counts[job_id] == 1:
            results[job_id] = result
    for job_id, count in counts.items():
        if count > 1:
            warnings.append(
                f"Duplicate research result records for job '{job_id}' cannot establish evidence."
            )
            results.pop(job_id, None)

    for day in plan.days:
        for activity in day.activities:
            if not _normalize_name(activity.name):
                issues.append(f"An activity on day {day.day_number} is missing its name.")
                continue
            job_id = activity.evidence_job_id
            if not job_id:
                issues.append(
                    f"Activity '{activity.name}' on day {day.day_number} has no evidence_job_id."
                )
                continue
            issues_add, warning_add = _validate_reference(
                state, job_id, "activities", day.destination,
                activity.name, jobs, results,
            )
            issues.extend(issues_add)
            warnings.extend(warning_add)

        if day.accommodation:
            job_id = day.accommodation_evidence_job_id
            if not job_id:
                issues.append(
                    f"Accommodation '{day.accommodation}' on day {day.day_number} "
                    "has no accommodation evidence_job_id."
                )
                continue
            issues_add, warning_add = _validate_reference(
                state, job_id, "accommodation", day.destination,
                day.accommodation, jobs, results,
            )
            issues.extend(issues_add)
            warnings.extend(warning_add)
        elif day.accommodation_evidence_job_id:
            issues.append(f"Day {day.day_number} has an accommodation evidence ID without a hotel name.")
    return issues, warnings


def _validate_reference(
    state: TripState,
    job_id: str,
    expected_agent: str,
    destination: str,
    entity_name: str,
    jobs: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    job = jobs.get(job_id)
    if job is None:
        issues.append(f"Unknown evidence job ID '{job_id}'.")
        return issues, warnings
    if job.get("agent") != expected_agent:
        issues.append(
            f"Evidence job '{job_id}' is a '{job.get('agent')}' job, "
            f"not an '{expected_agent}' job."
        )
    if _normalize_name(job.get("destination")) != _normalize_name(destination):
        issues.append(
            f"Evidence job '{job_id}' belongs to '{job.get('destination')}', "
            f"not destination '{destination}'."
        )

    result = results.get(job_id)
    if result is None:
        issues.append(
            f"Remove the unsupported recommendation '{entity_name}' citing '{job_id}'."
        )
        warnings.append(
            f"Research result for evidence job '{job_id}' is missing or duplicated."
        )
        return issues, warnings
    if result.get("status") != "completed":
        issues.append(f"Remove the unsupported recommendation '{entity_name}' citing '{job_id}'.")
        warnings.append(
            f"Research job '{job_id}' has status '{result.get('status')}' "
            "and cannot establish evidence."
        )
        return issues, warnings
    if (
        _normalize_name(result.get("destination")) != _normalize_name(job.get("destination"))
        or result.get("agent") != job.get("agent")
    ):
        issues.append(
            f"Research result metadata mismatches evidence job '{job_id}'."
        )
        return issues, warnings

    content = _result_content(result)
    records, structured = _structured_records(content)
    if not structured:
        warnings.append(
            f"Evidence job '{job_id}' contains only free-form or unstructured text; "
            f"'{entity_name}' cannot be verified deterministically."
        )
        return issues, warnings

    normalized_name = _normalize_name(entity_name)
    names = {_normalize_name(_entity_name(record)) for record in records}
    supported = normalized_name in names
    if not supported:
        issues.append(
            f"Recommended {expected_agent} entity '{entity_name}' is not present "
            f"in the complete structured evidence from job '{job_id}'."
        )
    if expected_agent == "accommodation":
        budget_issues = _validate_hotel_budget(state, entity_name, records, job)
        issues.extend(budget_issues)
    return issues, warnings


def _hotel_over_budget(state: TripState, hotel: dict[str, Any], job: dict[str, Any]) -> bool:
    intent = state.get("trip_intent", {})
    if not isinstance(intent, dict):
        intent = intent.model_dump()
    budget = job.get("arguments", {}).get("budget", intent.get("nightly_hotel_budget"))
    budget_currency = intent.get("currency") or "USD"
    price = next(
        (hotel[key] for key in ("nightly_price", "nightly_rate", "price", "rate")
         if hotel.get(key) is not None),
        None,
    )
    currency = hotel.get("currency") or hotel.get("price_currency")
    if isinstance(price, dict):
        currency = price.get("currency") or currency
        price = price.get("amount")
    return (
        isinstance(price, (int, float)) and not isinstance(price, bool)
        and isinstance(budget, (int, float)) and not isinstance(budget, bool)
        and isinstance(currency, str) and _normalize_name(currency) == _normalize_name(budget_currency)
        and price > budget >= 0
    )


def _validate_hotel_budget(
    state: TripState, hotel_name: str, records: list[dict[str, Any]], job: dict[str, Any],
) -> list[str]:
    if any(
        _normalize_name(_entity_name(record)) == _normalize_name(hotel_name)
        and _hotel_over_budget(state, record, job)
        for record in records
    ):
        return [f"Hotel '{hotel_name}' has a researched price exceeding the nightly budget."]
    return []


def _validate_research_use(
    state: TripState, plan: TripItinerary, context: dict[str, Any],
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    allocation = context.get("day_allocation")
    if not isinstance(allocation, dict):
        return issues, warnings
    without_lodging = {
        slot["day_number"] for slot in context.get("day_slots", [])
        if slot.get("lodging_night") is False
    }
    for day in plan.days:
        if day.day_number in without_lodging and day.accommodation:
            issues.append(f"Day {day.day_number} falls outside the lodging dates; leave both accommodation fields empty.")

    grouped = context.get("research_by_destination")
    if not isinstance(grouped, dict):
        if not state.get("trip_intent"):
            return issues, warnings
        grouped = group_research_by_destination(state)
    jobs = {
        job["job_id"]: job for job in state.get("validated_jobs", [])
        if isinstance(job, dict) and "job_id" in job
    }
    for item in context.get("research_coverage", {}).get("evidence_issues", []):
        if item.get("reason") == "assignment_mismatch":
            warnings.append(f"Research metadata for job '{item['job_id']}' did not match its assignment.")
    for destination, count in allocation.items():
        if type(count) is not int or count <= 0:
            continue
        destination_jobs = [job for job in jobs.values() if job.get("destination") == destination]
        evidence = grouped.get(destination, {})
        days = [day for day in plan.days if _normalize_name(day.destination) == _normalize_name(destination)]
        if any(job.get("agent") == "activities" for job in destination_jobs):
            names_by_job: dict[str, set[str]] = {}
            unstructured = False
            for result in evidence.get("activities", []):
                records, structured = _structured_records(_result_content(result))
                unstructured |= not structured
                names_by_job[result["job_id"]] = {
                    name for record in records if (name := _normalize_name(_entity_name(record)))
                }
            distinct = set().union(*names_by_job.values())
            target = min(len(distinct), count)
            if unstructured:
                warnings.append(f"{destination}: free-form attraction research could not be counted.")
            elif len(distinct) < count:
                warnings.append(
                    f"{destination}: {len(distinct)} distinct researched attractions for {count} itinerary days."
                )

            used: set[str] = set()
            occupied = 0
            for day in days:
                names = {
                    name for activity in day.activities
                    if (name := _normalize_name(activity.name)) in names_by_job.get(activity.evidence_job_id, set())
                }
                occupied += bool(names)
                used.update(names)
            if len(used) < target or occupied < target:
                issues.append(
                    f"{destination}: distribute at least {target} distinct researched attractions "
                    f"across {target} different days before adding extra stops."
                )

        candidates = []
        for result in evidence.get("accommodation", []):
            records, _ = _structured_records(_result_content(result))
            warnings.extend(lodging_limitations(records, destination))
            candidates.extend(
                record for record in records
                if _normalize_name(_entity_name(record))
                and not _hotel_over_budget(state, record, jobs.get(result["job_id"], {}))
            )
        eligible_days = [day for day in days if day.day_number not in without_lodging]
        hotels_selected = {
            _normalize_name(day.accommodation) for day in eligible_days if day.accommodation
        }
        if len(hotels_selected) > 1:
            issues.append(f"{destination}: choose one researched property for the validated lodging stay.")
        if candidates and eligible_days and not hotels_selected:
            issues.append(
                f"{destination}: include a researched lodging candidate and its evidence job ID."
            )
        elif not candidates and any(job.get("agent") == "accommodation" for job in destination_jobs):
            warnings.append(f"{destination}: no usable lodging candidate was established.")
    return issues, warnings


def determine_validation_status(
    issues: list[str],
    research_warnings: list[str],
) -> str:
    if issues:
        return "invalid"
    if research_warnings:
        return "partial"
    return "verified"


def create_validate_final_node(_context: TripNodeContext):
    async def validate_final(state: TripState) -> dict[str, Any]:
        plan, issues = validate_structure(state)
        coverage = state.get("research_coverage") or {}
        research_warnings = execution_warnings(coverage)
        updates: dict[str, Any] = {}

        if plan is not None:
            if not re.search(r"\b(?:repeat|revisit|return to|twice)\b", state.get("user_request", ""), re.I):
                scheduled = set()
                for day in plan.days:
                    distinct = []
                    for activity in day.activities:
                        key = (_normalize_name(day.destination), _normalize_name(activity.name), activity.evidence_job_id)
                        if key not in scheduled:
                            scheduled.add(key)
                            distinct.append(activity)
                    day.activities = distinct
            itinerary_context = state.get("itinerary_context")
            if not isinstance(itinerary_context, dict):
                issues.append("Itinerary context is missing or malformed.")
            else:
                issues.extend(validate_day_constraints(plan, itinerary_context))
                issues.extend(validate_destination_allocation(plan, itinerary_context))
                use_issues, use_warnings = _validate_research_use(state, plan, itinerary_context)
                issues.extend(use_issues)
                research_warnings.extend(use_warnings)

            evidence_issues, evidence_warnings = validate_evidence_references(state, plan)
            issues.extend(evidence_issues)
            research_warnings.extend(evidence_warnings)
            plan.warnings = sorted(set(research_warnings))
            previous = None
            for day in plan.days:
                day.transfer_notes = (
                    f"Transfer from {previous} to {day.destination}. Check transport options and journey time."
                    if previous is not None and _normalize_name(previous) != _normalize_name(day.destination)
                    else None
                )
                previous = day.destination
                for activity in day.activities:
                    activity.notes = None
            updates["structured_plan"] = plan.model_dump(mode="json")

        issues = sorted(set(issues))
        research_warnings = sorted(set(research_warnings))
        status = determine_validation_status(issues, research_warnings)
        logger.info(
            "Planning attempt=%d status=%s correctable_issues=%s research_limitations=%s",
            state.get("plan_attempts", 0), status, issues, research_warnings,
        )
        return {**updates, "issues": issues, "research_warnings": research_warnings, "status": status}

    return validate_final
