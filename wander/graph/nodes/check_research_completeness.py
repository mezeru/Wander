import logging
from typing import Any

from ..schemas import TripState


logger = logging.getLogger(__name__)


def execution_warnings(coverage: dict[str, Any]) -> list[str]:
    return [
        message.format(job_id)
        for kind, message in {
            "failed": "Research job '{}' failed; evidence is unavailable.",
            "no_data": "Research job '{}' returned no usable places.",
            "missing": "Research result for job '{}' is missing.",
            "duplicate": "Duplicate research results for job '{}' cannot establish evidence.",
            "unknown": "Unknown or malformed research result '{}' is not authoritative.",
        }.items()
        for job_id in coverage.get(kind, [])
    ]


def lodging_limitations(records: list[dict[str, Any]], destination: str) -> list[str]:
    warnings = []
    unknown_prices = sum(
        "nightly_price" in hotel and hotel["nightly_price"] is None
        and hotel.get("nightly_rate") is None
        for hotel in records
    )
    if unknown_prices:
        scope = "all" if unknown_prices == len(records) else "some"
        warnings.append(f"{destination} lodging: nightly prices and budget fit are unknown for {scope} candidates.")
    if any(hotel.get("availability_verified") is False for hotel in records):
        warnings.append(f"{destination} lodging: room availability was not verified.")
    if any(hotel.get("provider") == "mock" and (
        hotel.get("nightly_price") is not None or hotel.get("nightly_rate") is not None
    ) for hotel in records):
        warnings.append(f"{destination} lodging: sample prices are illustrative.")
    return warnings


def create_check_research_completeness_node(_context=None):
    def check_research_completeness(state: TripState) -> dict[str, Any]:
        expected_ids = []
        for job in state.get("validated_jobs", []):
            if isinstance(job, dict) and job.get("job_id"):
                expected_ids.append(str(job["job_id"]))
        expected_set = set(expected_ids)

        completed = []
        failed = []
        no_data = []
        duplicate = []
        unknown = []
        seen = set()

        for record in state.get("research_results", []):
            if not isinstance(record, dict):
                unknown.append("<malformed>")
                continue
            job_id = str(record.get("job_id") or "")
            if not job_id:
                unknown.append("<missing>")
                continue
            if job_id in seen:
                duplicate.append(job_id)
            seen.add(job_id)
            if job_id not in expected_set:
                unknown.append(job_id)
                continue

            status = record.get("status")
            if status == "completed":
                result = record.get("result")
                usable = (
                    isinstance(result, str) and bool(result.strip())
                    or isinstance(result, (dict, list)) and bool(result)
                )
                if usable:
                    completed.append(job_id)
                else:
                    unknown.append(job_id)
            elif status == "failed":
                failed.append(job_id)
            elif status == "no_data":
                no_data.append(job_id)
            else:
                unknown.append(job_id)

        unique_completed = sorted(set(completed))
        unique_failed = sorted(set(failed))
        unique_no_data = sorted(set(no_data))
        missing = sorted(
            job_id for job_id in expected_set
            if job_id not in set(unique_completed)
            and job_id not in set(unique_failed)
            and job_id not in set(unique_no_data)
        )

        coverage = {
            "expected": len(expected_set),
            "completed": len(unique_completed),
            "finished": len(expected_set & seen),
            "failed": unique_failed,
            "no_data": unique_no_data,
            "missing": missing,
            "duplicate": sorted(set(duplicate)),
            "unknown": sorted(set(unknown)),
        }
        logger.info(
            "Research batch: finished=%d/%d usable=%d failed=%d no_data=%d",
            coverage["finished"], coverage["expected"], coverage["completed"],
            len(unique_failed), len(unique_no_data),
        )
        warnings = execution_warnings(coverage)
        updates = {"research_coverage": coverage, "research_warnings": warnings}
        if not (coverage["completed"] or coverage["no_data"]):
            updates.update(status="research_failed", status_message="No research job produced usable evidence.")
        elif state.get("trip_intent", {}).get("requires_itinerary") is False:
            for result in state.get("research_results", []):
                report = result.get("result") if isinstance(result, dict) else None
                if isinstance(report, dict) and isinstance(report.get("hotels"), list):
                    warnings.extend(lodging_limitations(report["hotels"], result["destination"]))
            updates["status"] = "partial" if warnings else "research_complete"
        return updates

    return check_research_completeness
