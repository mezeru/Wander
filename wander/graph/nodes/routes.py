import logging

from langgraph.types import Send

from ..schemas import TripState


MAX_PLAN_ATTEMPTS = 3
MAX_CLARIFICATION_ROUNDS = 5
logger = logging.getLogger(__name__)


def dispatch_research(state: TripState) -> list[Send]:

    if state.get("missing_details"):
        raise ValueError("Cannot dispatch unvalidated jobs")

    jobs = state.get("validated_jobs", [])
    
    if not jobs:
        raise ValueError("No validated jobs available")

    sends = []

    for job in jobs:
        sends.append(
            Send(
                "research_worker",
                {
                    "job_id": job["job_id"],
                    "agent": job["agent"],
                    "destination": job["destination"],
                    "task": job["task"],
                    "arguments": job["arguments"],
                }
            )
        )
    return sends


def route_requirements(state: TripState) -> str:
    research_jobs = state.get("research_jobs", [])
    if not research_jobs:
        logger.info(
            "Workflow route: requirements_check -> needs_input "
            "(no research jobs generated)"
        )
        return "needs_input"
    if not state.get("missing_details"):
        logger.info("Workflow route: requirements_check -> ready_for_dispatch")
        return "ready_for_dispatch"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        logger.info(
            "Workflow route: requirements_check -> needs_input "
            "(clarification limit reached)"
        )
        return "needs_input"
    logger.info("Workflow route: requirements_check -> ask_user")
    return "ask_user"


def route_research_completeness(state: TripState) -> str:
    coverage = state.get("research_coverage", {})
    integrity = (
        coverage.get("missing", [])
        or coverage.get("duplicate", [])
        or coverage.get("unknown", [])
    )
    if coverage.get("completed", 0) == 0:
        logger.info(
            "Workflow route: check_research_completeness -> needs_input "
            "(no usable research)"
        )
        return "needs_input"
    if integrity:
        logger.info(
            "Workflow route: check_research_completeness -> prepare_itinerary_context "
            "(partial coverage or orchestration integrity issue)"
        )
        return "prepare_itinerary_context"
    if coverage.get("failed") or coverage.get("no_data"):
        logger.info(
            "Workflow route: check_research_completeness -> prepare_itinerary_context "
            "(partial research coverage)"
        )
        return "prepare_itinerary_context"
    logger.info(
        "Workflow route: check_research_completeness -> prepare_itinerary_context "
        "(complete research coverage)"
    )
    return "prepare_itinerary_context"


def route_review(state: TripState) -> str:
    if state.get("status") == "partial":
        logger.info("Workflow route: validate_final -> END (partial research)")
        return "__end__"
    if state.get("status") == "verified" and not state.get("issues"):
        logger.info("Workflow route: validate_final -> END (verified)")
        return "__end__"
    if state.get("plan_attempts", 0) < MAX_PLAN_ATTEMPTS:
        logger.info(
            "Workflow route: validate_final -> final_planner "
            "(revision required)"
        )
        return "review_again"
    logger.info("Workflow route: validate_final -> END (attempt limit reached)")
    return "__end__"
