import logging
from typing import Any

from ..schemas import ResearchJob, TripState
from .context import TripNodeContext


logger = logging.getLogger(__name__)


def create_research_jobs_node(_context: TripNodeContext):
    async def create_research_jobs(state: TripState) -> dict[str, Any]:
        intent = state["trip_intent"]
        destinations = intent["destinations"]
        jobs = []
        for index, destination in enumerate(destinations):
            if intent["needs_activities"]:
                jobs.append(ResearchJob(
                    job_id=f"activities-{index}", agent="activities",
                    destination=destination, tool_name="search_attractions",
                    task=f"Research attractions in {destination}",
                    arguments={"destination": destination},
                ))

            if intent["needs_accommodation"]:
                arguments: dict[str, Any] = {"destination": destination}
                if intent.get("nightly_hotel_budget") is not None:
                    arguments["budget"] = intent["nightly_hotel_budget"]
                for field in ("check_in", "check_out"):
                    dates = {
                        stay[field] for stay in intent.get("hotel_stays", [])
                        if stay.get(field) and (
                            (stay.get("destination") or "").strip().casefold() == destination.casefold()
                            or (not stay.get("destination") and len(destinations) == 1)
                        )
                    }
                    if len(dates) == 1:
                        arguments[field] = dates.pop()
                jobs.append(ResearchJob(
                    job_id=f"accommodation-{index}", agent="accommodation",
                    destination=destination, tool_name="search_hotels",
                    task=f"Research hotels in {destination}", arguments=arguments,
                ))

        logger.info("Created %d research jobs: %s", len(jobs),
                    [(job.job_id, job.destination, job.agent) for job in jobs])
        return {"research_jobs": [job.model_dump() for job in jobs]}

    return create_research_jobs
