from typing import Any

from langgraph.types import interrupt

from ..schemas import TripState
from ..validation import normalize_value
from .context import TripNodeContext


def create_ask_user_node(context: TripNodeContext):
    def ask_user(state: TripState) -> dict[str, Any]:
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
        for job_id, fields in answers.items():
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

        return {
            "job_answers": job_answers,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }

    return ask_user
