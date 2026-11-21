import json
from typing import Any
from pydantic import ValidationError

from ..schemas import TripItinerary, TripState
from .context import TripNodeContext


def _normalize_name(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def validate_structure(state: TripState) -> tuple[TripItinerary | None, list[str]]:
    raw = state.get("structured_plan")
    if raw is None:
        return None, ["Structured itinerary is missing."]
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


def _structured_records(content: Any, agent: str) -> tuple[list[dict[str, Any]], bool]:
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


def _entity_id(record: dict[str, Any]) -> str:
    for key in ("id", "entity_id", "attraction_id", "hotel_id"):
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
    for job in state.get("validated_jobs", []) or []:
        if not isinstance(job, dict):
            warnings.append("Research contains a malformed validated job.")
            continue
        job_id = job.get("job_id")
        if isinstance(job_id, str) and job_id:
            if job_id in jobs:
                issues.append(f"Duplicate authoritative research job ID '{job_id}'.")
            else:
                jobs[job_id] = job

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
        warnings.append(
            f"Research result for evidence job '{job_id}' is missing or duplicated."
        )
        return issues, warnings
    if result.get("status") != "completed":
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
    records, structured = _structured_records(content, expected_agent)
    if not structured:
        warnings.append(
            f"Evidence job '{job_id}' contains only free-form or unstructured text; "
            f"'{entity_name}' cannot be verified deterministically."
        )
        return issues, warnings

    recommended_id = ""
    normalized_name = _normalize_name(entity_name)
    names = {_normalize_name(_entity_name(record)) for record in records}
    supported = normalized_name in names
    if not supported:
        issues.append(
            f"Recommended {expected_agent} entity '{entity_name}' is not present "
            f"in the complete structured evidence from job '{job_id}'."
        )
    if expected_agent == "accommodation":
        budget_issues = _validate_hotel_budget(state, entity_name, records)
        issues.extend(budget_issues)
    return issues, warnings


def _validate_hotel_budget(
    state: TripState,
    hotel_name: str,
    records: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []
    trip_intent = state.get("trip_intent")
    budget: Any = None
    budget_currency: str | None = None
    if isinstance(trip_intent, dict):
        budget = trip_intent.get("nightly_hotel_budget")
        budget_currency = trip_intent.get("currency")
    elif hasattr(trip_intent, "nightly_hotel_budget"):
        budget = getattr(trip_intent, "nightly_hotel_budget")
        budget_currency = getattr(trip_intent, "currency", None)

    for job_id_result in state.get("research_results", []) or []:
        if not isinstance(job_id_result, dict):
            continue
        # Job arguments are not included in result records; fallback fields may
        # provide explicit currency/budget metadata without fabrication.
        if isinstance(job_id_result.get("currency"), str):
            budget_currency = budget_currency or job_id_result["currency"]
        if job_id_result.get("budget") is not None:
            budget = budget if budget is not None else job_id_result["budget"]

    tool_arguments = state.get("tool_arguments")
    if isinstance(tool_arguments, dict):
        hotel_arguments = tool_arguments.get("search_hotels")
        if isinstance(hotel_arguments, dict):
            if budget is None:
                budget = hotel_arguments.get("budget")
            if budget_currency is None:
                budget_currency = hotel_arguments.get("currency")

    user_context = state.get("user_context")
    if isinstance(user_context, dict):
        for values in user_context.values():
            if not isinstance(values, dict):
                continue
            if budget is None:
                budget = values.get("nightly_hotel_budget", values.get("nightly_budget"))
            if budget_currency is None:
                budget_currency = values.get("currency")

    for record in records:
        if _normalize_name(_entity_name(record)) != _normalize_name(hotel_name):
            continue
        price = next(
            (
                record[key]
                for key in ("nightly_rate", "nightly_price", "price", "rate")
                if key in record
            ),
            None,
        )
        currency = next(
            (
                record[key]
                for key in ("currency", "price_currency")
                if isinstance(record.get(key), str)
            ),
            None,
        )
        if (
            isinstance(price, (int, float))
            and not isinstance(price, bool)
            and price >= 0
            and isinstance(budget, (int, float))
            and not isinstance(budget, bool)
            and budget >= 0
            and isinstance(currency, str)
            and isinstance(budget_currency, str)
            and currency.strip().casefold() == budget_currency.strip().casefold()
        ):
            if float(price) > float(budget):
                issues.append(
                    f"Hotel '{hotel_name}' costs {price} {currency}, "
                    f"exceeding the {budget} {budget_currency} nightly budget."
                )
    return issues


def collect_research_warnings(
    state: TripState,
    evidence_warnings: list[str],
) -> list[str]:
    warnings = list(evidence_warnings)
    coverage = state.get("research_coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    for job_id in coverage.get("failed", []) or []:
        warnings.append(f"Research job '{job_id}' failed; evidence is unavailable.")
    for job_id in coverage.get("no_data", []) or []:
        warnings.append(f"Research job '{job_id}' returned no data.")
    for job_id in coverage.get("missing", []) or []:
        warnings.append(f"Research result for job '{job_id}' is missing.")
    for job_id in coverage.get("duplicate", []) or []:
        warnings.append(
            f"Duplicate research result records for job '{job_id}' "
            "cannot establish evidence."
        )
    for job_id in coverage.get("unknown", []) or []:
        warnings.append(f"Unknown research result '{job_id}' is not authoritative.")
    return sorted(set(warning for warning in warnings if warning))


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
        if plan is None:
            status = determine_validation_status(issues, [])
            return {"issues": sorted(set(issues)), "research_warnings": [], "status": status}

        itinerary_context = state.get("itinerary_context")
        if not isinstance(itinerary_context, dict):
            issues.append("Itinerary context is missing or malformed.")
        else:
            issues.extend(validate_day_constraints(plan, itinerary_context))
            issues.extend(validate_destination_allocation(plan, itinerary_context))

        evidence_issues, evidence_warnings = validate_evidence_references(state, plan)
        issues.extend(evidence_issues)
        research_warnings = collect_research_warnings(state, evidence_warnings)
        issues = sorted(set(issues))
        research_warnings = sorted(set(research_warnings))
        status = determine_validation_status(issues, research_warnings)
        return {
            "issues": issues,
            "research_warnings": research_warnings,
            "status": status,
        }

    return validate_final
