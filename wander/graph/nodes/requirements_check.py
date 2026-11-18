from typing import Any

from ..schemas import TripState
from ..validation import validate_arguments
from .context import TripNodeContext


def create_requirements_check_node(context: TripNodeContext):
    def requirements_check(state: TripState) -> dict[str, Any]:
        selected_schemas = {}
        if state.get("activities_task"):
            selected_schemas["search_attractions"] = context.schemas[
                "search_attractions"
            ]
        if state.get("accommodation_task"):
            selected_schemas["search_hotels"] = context.schemas["search_hotels"]

        checked, issues = validate_arguments(
            state.get("tool_arguments", {}),
            selected_schemas,
            state["user_request"],
            state.get("user_context", {}),
            state.get("input_errors", []),
        )
        return {
            "tool_arguments": {
                **state.get("tool_arguments", {}),
                **checked,
            },
            "missing_details": issues,
        }

    return requirements_check
