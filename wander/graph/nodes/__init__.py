from typing import Any, Callable

from ..logging import log_node
from .accommodation import create_accommodation_node
from .activities import create_activities_node
from .ask_user import create_ask_user_node
from .context import TripNodeContext
from .coordinator import create_coordinator_node
from .final_planner import create_final_planner_node
from .needs_input import needs_input
from .requirements_check import create_requirements_check_node
from .research_worker import create_research_worker_node
from .route_agents import route_agents
from .trip_intent import create_extract_trip_intent_node
from .validate_final import create_validate_final_node


def build_nodes(context: TripNodeContext) -> dict[str, Callable[..., Any]]:
    return {
        "extract_trip_intent": log_node(
            "extract_trip_intent",
            create_extract_trip_intent_node(context),
        ),
        "coordinator": log_node(
            "coordinator",
            create_coordinator_node(context),
        ),
        "requirements_check": log_node(
            "requirements_check",
            create_requirements_check_node(context),
        ),
        "route_agents": log_node("route_agents", route_agents),
        "research_worker": log_node(
            "research_worker",
            create_research_worker_node(context),
        ),
        "ask_user": log_node("ask_user", create_ask_user_node(context)),
        "needs_input": log_node("needs_input", needs_input),
        "activities": log_node(
            "activities",
            create_activities_node(context),
        ),
        "accommodation": log_node(
            "accommodation",
            create_accommodation_node(context),
        ),
        "final_planner": log_node(
            "final_planner",
            create_final_planner_node(context),
        ),
        "validate_final": log_node(
            "validate_final",
            create_validate_final_node(context),
        ),
    }


__all__ = ["TripNodeContext", "build_nodes"]
