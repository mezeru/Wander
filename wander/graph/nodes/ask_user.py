from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from langgraph.types import interrupt

from ..schemas import TripState
from ..validation import issue, missing, normalize_value
from .context import TripNodeContext


def create_ask_user_node(context: TripNodeContext):
    def ask_user(state: TripState) -> dict[str, Any]:
        answers = interrupt(
            {
                "type": "missing_details",
                "fields": state["missing_details"],
                "message": "Please provide the requested details so research can continue.",
            }
        )
        if not isinstance(answers, dict):
            raise ValueError("Expected answers grouped by tool name.")
        user_context = {
            name: dict(fields)
            for name, fields in state.get("user_context", {}).items()
        }
        errors_by_key = {
            (entry["tool"], entry["field"]): entry
            for entry in state.get("input_errors", [])
        }
        for name, fields in answers.items():
            if name not in context.schemas and name != "__trip__":
                continue
            if not isinstance(fields, dict):
                continue
            for field, raw_value in fields.items():
                key = (name, field)
                try:
                    if name == "__trip__" and field == "confirm_extra_nights":
                        value = normalize_value(raw_value, {"type": "boolean"}, field)
                        if value is False:
                            errors_by_key[("search_hotels", "check_out")] = issue(
                                "search_hotels",
                                "check_out",
                                "Enter corrected check-out date for your trip.",
                            )
                        user_context.setdefault(name, {})[field] = value
                    elif (
                        name in context.schemas
                        and field in context.schemas[name].get("properties", {})
                    ):
                        prop = context.schemas[name]["properties"][field]
                        value = normalize_value(raw_value, prop, field)
                        if missing(value):
                            raise ValueError("A value is required.")
                        Draft202012Validator(
                            {
                                "type": "object",
                                "properties": {field: prop},
                                "$defs": context.schemas[name].get("$defs", {}),
                            },
                            format_checker=FormatChecker(),
                        ).validate({field: value})
                        user_context.setdefault(name, {})[field] = value
                    else:
                        continue
                    errors_by_key.pop(key, None)
                except (ValueError, TypeError, ValidationError) as exc:
                    detail = exc.message if isinstance(exc, ValidationError) else str(exc)
                    errors_by_key[key] = issue(
                        name, field, f"Invalid {field}: {detail}"
                    )
        return {
            "user_context": user_context,
            "input_errors": list(errors_by_key.values()),
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        }

    return ask_user
