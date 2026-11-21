from ..schemas import TripState, ResearchJob
from .context import TripNodeContext

import logging
from typing import Any

logger = logging.getLogger(__name__)

def create_research_jobs_node(context: TripNodeContext):
    async def create_research_jobs(state: TripState):
        intent = state["trip_intent"]
        jobs = []

        for index, destination in enumerate(intent["destinations"]):

            if intent["needs_activities"]:
                jobs.append(
                    ResearchJob(
                        job_id=f"activities-{index}",
                        agent="activities",
                        destination=destination,
                        tool_name="search_attractions",
                        task=f"Research attractions in {destination}",
                        arguments={
                            "destination": destination
                        }
                    )
                )

            if intent["needs_accommodation"]:
                hotel_args = {
                    "destination": destination
                }

                budget = intent.get("nightly_hotel_budget")

                if budget is not None:
                    hotel_args["budget"] = budget

                jobs.append(
                    ResearchJob(
                        job_id=f"accommodation-{index}",
                        agent="accommodation",
                        destination=destination,
                        tool_name="search_hotels",
                        task=f"Research hotels in {destination}",
                        arguments=hotel_args
                    )
                )

                logger.info(f"The jobs created from intent {jobs}")

        return {
            "research_jobs": [
                job.model_dump()
                for job in jobs
            ]
        }

    return create_research_jobs