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
                    "tool_name": job["tool_name"],
                    "user_request": state.get("user_request", ""),
                }
            )
        )
    return sends


def route_requirements(state: TripState) -> str:
    if state.get("status") in {"workflow_error", "no_research"}:
        route = "__end__"
    elif state.get("missing_details"):
        route = (
            "needs_input" if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS
            else "ask_user"
        )
    elif "research_jobs" not in state:
        route = "create_research_jobs"
    elif state.get("validated_jobs"):
        route = "ready_for_dispatch"
    else:
        route = "__end__"
    logger.info("Workflow route: requirements_check -> %s", route)
    return route


def route_research_completeness(state: TripState) -> str:
    coverage = state.get("research_coverage", {})
    if not (coverage.get("completed") or coverage.get("no_data")):
        route = "__end__"
    elif state.get("trip_intent", {}).get("requires_itinerary") is False:
        route = "__end__"
    else:
        route = "prepare_itinerary_context"
    logger.info("Workflow route: check_research_completeness -> %s", route)
    return route


def route_review(state: TripState) -> str:
    if state.get("status") == "invalid" and state.get("issues"):
        if state.get("plan_attempts", 0) < MAX_PLAN_ATTEMPTS:
            logger.info(
                "Workflow route: validate_final -> final_planner "
                "(correctable issues remain)"
            )
            return "review_again"
        logger.info(
            "Workflow route: validate_final -> END "
            "(attempt limit reached with unresolved issues)"
        )
        return "__end__"
    if state.get("status") in ("verified", "partial"):
        logger.info(
            "Workflow route: validate_final -> END (verified)"
            if state.get("status") == "verified"
            else "Workflow route: validate_final -> END (partial research)"
        )
        return "__end__"
    logger.info("Workflow route: validate_final -> END")
    return "__end__"
