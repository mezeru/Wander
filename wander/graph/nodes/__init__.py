from typing import Any, Callable

from ..logging import log_node
from .ask_user import create_ask_user_node
from .context import TripNodeContext
from .final_planner import create_final_planner_node
from .needs_input import needs_input
from .requirements_check import create_requirements_check_node
from .research_worker import create_research_worker_node
from .route_agents import ready_for_dispatch
from .trip_intent import create_extract_trip_intent_node
from .validate_final import create_validate_final_node
from .job_creation import create_research_jobs_node
from .check_research_completeness import create_check_research_completeness_node
from .itinerary_context import prepare_itinerary_context


def build_nodes(context: TripNodeContext) -> dict[str, Callable[..., Any]]:
    return {
        "extract_trip_intent": log_node(
            "extract_trip_intent",
            create_extract_trip_intent_node(context),
        ),
        "requirements_check": log_node(
            "requirements_check",
            create_requirements_check_node(context),
        ),
        "ready_for_dispatch": log_node(
            "ready_for_dispatch",
            ready_for_dispatch,
        ),
        "research_worker": log_node(
            "research_worker",
            create_research_worker_node(context),
        ),
        "ask_user": log_node("ask_user", create_ask_user_node(context)),
        "needs_input": log_node("needs_input", needs_input),
        "final_planner": log_node(
            "final_planner",
            create_final_planner_node(context),
        ),
        "validate_final": log_node(
            "validate_final",
            create_validate_final_node(context),
        ),
        "create_research_jobs": log_node(
            "create_research_jobs",
            create_research_jobs_node(context),
        ),
        "check_research_completeness": log_node(
            "check_research_completeness",
            create_check_research_completeness_node(context),
        ),
        "prepare_itinerary_context": log_node(
            "prepare_itinerary_context",
            prepare_itinerary_context,
        ),
    }


__all__ = ["TripNodeContext", "build_nodes"]
