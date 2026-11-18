import logging

from langgraph.types import Send

from ..schemas import TripState


MAX_PLAN_ATTEMPTS = 3
MAX_CLARIFICATION_ROUNDS = 5
logger = logging.getLogger(__name__)


def dispatch_research(state: TripState) -> list[Send]:
    sends = []
    agents = []
    if state.get("activities_task"):
        agents.append("activities")
        sends.append(
            Send(
                "research_worker",
                {
                    "agent": "activities",
                    "task": state["activities_task"],
                    "arguments": state.get("tool_arguments", {}).get(
                        "search_attractions", {}
                    ),
                },
            )
        )
    if state.get("accommodation_task"):
        agents.append("accommodation")
        sends.append(
            Send(
                "research_worker",
                {
                    "agent": "accommodation",
                    "task": state["accommodation_task"],
                    "arguments": state.get("tool_arguments", {}).get(
                        "search_hotels", {}
                    ),
                },
            )
        )
    logger.info(
        "Workflow route: route_agents -> research_worker (agents=%s)",
        ", ".join(agents) if agents else "none",
    )
    return sends


def route_requirements(state: TripState) -> str:
    if not state.get("activities_task") and not state.get("accommodation_task"):
        logger.info(
            "Workflow route: requirements_check -> needs_input "
            "(no research tasks selected)"
        )
        return "needs_input"
    if not state.get("missing_details"):
        logger.info("Workflow route: requirements_check -> route_agents")
        return "route_agents"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        logger.info(
            "Workflow route: requirements_check -> needs_input "
            "(clarification limit reached)"
        )
        return "needs_input"
    logger.info("Workflow route: requirements_check -> ask_user")
    return "ask_user"


def route_review(state: TripState) -> str:
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
