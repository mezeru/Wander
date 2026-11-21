from typing import Any

from ..schemas import TripState


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
            "failed": unique_failed,
            "no_data": unique_no_data,
            "missing": missing,
            "duplicate": sorted(set(duplicate)),
            "unknown": sorted(set(unknown)),
        }
        return {"research_coverage": coverage}

    return check_research_completeness
