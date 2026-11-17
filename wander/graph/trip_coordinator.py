from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .accommodation_agent import build_accommodation_graph
from .activities_graph import build_activities_graph


MAX_PLAN_ATTEMPTS = 3
MAX_CLARIFICATION_ROUNDS = 5
MODEL_NAME = "qwen3:4b"


class ResearchTasks(BaseModel):
    activities_task: str = Field(description="Instructions for the Activities Agent")
    accommodation_task: str = Field(description="Instructions for the Accommodation Agent")
    tool_arguments: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Arguments grouped by MCP tool name; omit unknown values.",
    )


class ValidationResult(BaseModel):
    issues: list[str] = Field(default_factory=list)
    status: Literal["verified", "needs_revision"]


class TripState(TypedDict):
    user_request: str
    activities_task: NotRequired[str]
    accommodation_task: NotRequired[str]
    activities_result: NotRequired[str]
    accommodation_result: NotRequired[str]
    tool_arguments: NotRequired[dict[str, dict[str, Any]]]
    user_context: NotRequired[dict[str, dict[str, Any]]]
    missing_details: NotRequired[list[dict[str, str]]]
    input_errors: NotRequired[list[dict[str, str]]]
    clarification_rounds: NotRequired[int]
    final_plan: NotRequired[str]
    plan_attempts: NotRequired[int]
    issues: NotRequired[list[str]]
    status: NotRequired[Literal["pending", "needs_input", "verified", "needs_revision"]]


def tool_schema(tool: Any) -> dict[str, Any]:
    """Read the LangChain schema adapted from MCP discovery."""
    schema = tool.tool_call_schema
    return schema if isinstance(schema, dict) else schema.model_json_schema()


def missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def parse_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a date, including the year (YYYY-MM-DD preferred).")
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError("Use YYYY-MM-DD or MM/DD/YYYY with a four-digit year.")


def parse_number(value: Any, *, integer: bool, field: str) -> int | float:
    if isinstance(value, bool):
        raise ValueError("Enter a number, not a boolean.")
    if isinstance(value, (int, float)):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if integer:
            if not re.fullmatch(r"[+-]?\d+", text):
                raise ValueError("Enter a whole number (e.g. 5).")
            number = int(text)
        else:
            # We accept USD notation, but do not silently convert EUR/GBP to USD.
            if any(symbol in text for symbol in ("€", "£", "¥")) or re.search(
                r"\b(?:EUR|GBP|JPY|INR|CAD|AUD)\b", text, re.IGNORECASE
            ):
                raise ValueError("Currency is unclear; enter a USD amount such as $150.")
            text = re.sub(r"(?i)\bUSD\b", "", text).strip()
            match = re.fullmatch(r"\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\$?", text)
            if not match:
                raise ValueError("Enter a numeric amount (e.g. 150 or $150).")
            number = float(match.group(1).replace(",", ""))
    else:
        raise ValueError("Enter a numeric value.")
    if not integer and not float("-inf") < float(number) < float("inf"):
        raise ValueError("Enter a finite number.")
    if field in {"budget", "nightly_rate", "price", "limit"} and number <= 0:
        raise ValueError("This value must be greater than zero.")
    return int(number) if integer else float(number)


def normalize_value(value: Any, field_schema: dict[str, Any], field: str) -> Any:
    """Normalize CLI/LLM inputs to the type advertised by MCP."""
    if missing(value):
        return None
    choices = field_schema.get("anyOf") or field_schema.get("oneOf")
    if choices:
        branch = next((part for part in choices if part.get("type") != "null"), {})
        field_schema = {**field_schema, **branch}

    kind = field_schema.get("type")
    if field_schema.get("format") == "date" or field in {"check_in", "check_out"}:
        return parse_date(value)
    if kind == "array":
        if isinstance(value, str):
            values = [part.strip() for part in re.split(r"[,;]|\band\b", value) if part.strip()]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            raise ValueError("Enter a comma-separated list.")
        item_schema = field_schema.get("items", {})
        return [normalize_value(item, item_schema, field) for item in values]
    if kind == "integer":
        return parse_number(value, integer=True, field=field)
    if kind == "number":
        return parse_number(value, integer=False, field=field)
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"yes", "y", "true", "1"}:
            return True
        if str(value).strip().lower() in {"no", "n", "false", "0"}:
            return False
        raise ValueError("Answer yes or no.")
    if kind == "string" or kind is None:
        if not isinstance(value, str):
            raise ValueError("Enter text.")
        return value.strip()
    return value


def explicit_nightly_budget(text: str) -> float | None:
    """Only infer hotel budget from clearly stated per-night language."""
    patterns = (
        r"(?:nightly(?:\s+hotel)?\s+budget|hotel\s+budget\s+per\s+night|nightly\s+(?:hotel\s+)?limit)\s*(?:of|is|:)?\s*(\$?\s*[\d,]+(?:\.\d+)?)",
        r"(?:hotel|room)\s+(?:budget|limit|rate)\s*(?:of|is|:)?\s*(\$?\s*[\d,]+(?:\.\d+)?)\s*(?:per\s+night|/\s*night)",
        r"(\$?\s*[\d,]+(?:\.\d+)?)\s*(?:per\s+night|/\s*night)\s*(?:hotel|room)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(parse_number(match.group(1), integer=False, field="budget"))
    return None


def issue(tool: str, field: str, description: str) -> dict[str, str]:
    return {"tool": tool, "field": field, "description": description}


def dedupe_issues(items: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for item in items:
        by_key[(item["tool"], item["field"])] = item
    return list(by_key.values())


def validate_arguments(
    arguments: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    user_request: str,
    context: dict[str, dict[str, Any]],
    input_errors: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Normalize, schema-validate, and check basic trip constraints."""
    normalized: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = list(input_errors)

    for tool_name, schema in specs.items():
        properties = schema.get("properties", {})
        provided = arguments.get(tool_name, {})
        valid: dict[str, Any] = {}
        for field, value in provided.items():
            if field not in properties or missing(value):
                continue
            try:
                converted = normalize_value(value, properties[field], field)
                Draft202012Validator(
                    {"type": "object", "properties": {field: properties[field]}, "$defs": schema.get("$defs", {})},
                    format_checker=FormatChecker(),
                ).validate({field: converted})
                valid[field] = converted
            except (ValueError, TypeError, ValidationError) as exc:
                detail = exc.message if isinstance(exc, ValidationError) else str(exc)
                issues.append(issue(tool_name, field, f"Invalid {field}: {detail}"))
        for field in schema.get("required", []):
            if field not in valid:
                issues.append(issue(tool_name, field, properties.get(field, {}).get("description") or f"Please provide {field}."))
        normalized[tool_name] = valid

    hotel = normalized.get("search_hotels", {})
    start, end = hotel.get("check_in"), hotel.get("check_out")
    if start and end:
        nights = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if nights <= 0:
            issues.append(issue("search_hotels", "check_out", "Check-out must be later than check-in."))
        else:
            match = re.search(r"\b(\d+)\s*[- ]?days?\b", user_request, re.IGNORECASE)
            if match and nights > int(match.group(1)) and not context.get("__trip__", {}).get("confirm_extra_nights"):
                issues.append(issue("__trip__", "confirm_extra_nights", f"The stay has {nights} hotel nights, but the trip says {match.group(1)} days. Accept the extra nights? (yes/no)"))

    return normalized, dedupe_issues(issues)


async def main() -> None:
    load_dotenv()
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    coordinator_model = model.with_structured_output(ResearchTasks)
    validator_model = model.with_structured_output(ValidationResult)

    # Langfuse is optional; tracing should not block local experiments.
    callbacks: list[Any] = []
    langfuse_client = None
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        langfuse_client = get_client()
        callbacks.append(CallbackHandler())
    except ImportError:
        pass

    server_path = Path(__file__).resolve().parent.parent / "mcp_server.py"
    async with MCPAdapter(server_path) as adapter:
        discovered = {tool.name: tool for tool in await adapter.list_tools()}
        needed = {name: discovered[name] for name in ("search_attractions", "search_hotels") if name in discovered}
        if len(needed) != 2:
            raise RuntimeError("The MCP server must expose search_attractions and search_hotels.")
        schemas = {name: tool_schema(tool) for name, tool in needed.items()}
        tool_specs = [
            {"name": name, "description": tool.description, "schema": schemas[name]}
            for name, tool in needed.items()
        ]

        # Existing builders are async in your project. Keep MCPAdapter open
        # throughout the entire graph invocation (including interrupt/resume).
        activities_graph = await build_activities_graph([needed["search_attractions"]])
        accommodation_graph = await build_accommodation_graph([needed["search_hotels"]])

        async def coordinator(state: TripState) -> dict[str, Any]:
            explicit_context = state.get("user_context", {})
            response = await coordinator_model.ainvoke([
                SystemMessage(content=(
                    "You coordinate WanderMind research. Create an activities task and an accommodation task. "
                    "Fill tool_arguments using the EXACT parameter names in the discovered schemas. "
                    "Extract all explicitly stated destinations, dates, interests, and NIGHTLY hotel budgets. "
                    "A total trip budget is NOT a nightly hotel budget. If that is all the user provided, "
                    "omit search_hotels.budget and ask for a nightly amount. "
                    "Never guess dates, guests, budget, or availability. Omit values you do not know. "
                    "Use user corrections as authoritative. Keep specialist tasks consistent with the details."
                )),
                HumanMessage(content=(
                    f"Original request:\n{state['user_request']}\n\n"
                    f"Latest validated user clarifications:\n{json.dumps(explicit_context)}\n\n"
                    f"Available tools:\n{json.dumps(tool_specs)}"
                )),
            ])
            if not isinstance(response, ResearchTasks):
                raise RuntimeError("Coordinator did not return validated ResearchTasks.")

            args = {name: dict(response.tool_arguments.get(name) or {}) for name in schemas}
            nightly = explicit_nightly_budget(state["user_request"])
            if nightly is not None:
                args["search_hotels"]["budget"] = nightly
            elif re.search(r"\b(?:total|overall|entire)\s+(?:trip\s+)?budget\b", state["user_request"], re.IGNORECASE):
                # Do not let a guessed nightly budget sneak in from the LLM.
                args["search_hotels"].pop("budget", None)

            # Latest explicit user clarifications always win over inferred values.
            for name, fields in explicit_context.items():
                if name in args:
                    args[name].update(fields)

            return {
                "activities_task": response.activities_task,
                "accommodation_task": response.accommodation_task,
                "tool_arguments": args,
            }

        def requirements_check(state: TripState) -> dict[str, Any]:
            checked, issues = validate_arguments(
                state.get("tool_arguments", {}),
                schemas,
                state["user_request"],
                state.get("user_context", {}),
                state.get("input_errors", []),
            )
            return {"tool_arguments": checked, "missing_details": issues}

        def ask_user(state: TripState) -> dict[str, Any]:
            answers = interrupt({
                "type": "missing_details",
                "fields": state["missing_details"],
                "message": "Please provide the requested details so research can continue.",
            })
            if not isinstance(answers, dict):
                raise ValueError("Expected answers grouped by tool name.")
            context = {name: dict(fields) for name, fields in state.get("user_context", {}).items()}
            errors_by_key = {
                (entry["tool"], entry["field"]): entry
                for entry in state.get("input_errors", [])
            }
            for name, fields in answers.items():
                if name not in schemas and name != "__trip__":
                    continue
                if not isinstance(fields, dict):
                    continue
                for field, raw_value in fields.items():
                    key = (name, field)
                    try:
                        if name == "__trip__" and field == "confirm_extra_nights":
                            value = normalize_value(raw_value, {"type": "boolean"}, field)
                            if value is False:
                                # User rejected extra nights: request corrected dates.
                                errors_by_key[("search_hotels", "check_out")] = issue(
                                    "search_hotels", "check_out", "Enter corrected check-out date for your trip."
                                )
                            context.setdefault(name, {})[field] = value
                        elif name in schemas and field in schemas[name].get("properties", {}):
                            prop = schemas[name]["properties"][field]
                            value = normalize_value(raw_value, prop, field)
                            if missing(value):
                                raise ValueError("A value is required.")
                            Draft202012Validator(
                                {"type": "object", "properties": {field: prop}, "$defs": schemas[name].get("$defs", {})},
                                format_checker=FormatChecker(),
                            ).validate({field: value})
                            context.setdefault(name, {})[field] = value
                        else:
                            continue
                        errors_by_key.pop(key, None)
                    except (ValueError, TypeError, ValidationError) as exc:
                        detail = exc.message if isinstance(exc, ValidationError) else str(exc)
                        errors_by_key[key] = issue(name, field, f"Invalid {field}: {detail}")
            return {
                "user_context": context,
                "input_errors": list(errors_by_key.values()),
                "clarification_rounds": state.get("clarification_rounds", 0) + 1,
            }

        async def activities(state: TripState) -> dict[str, str]:
            args = state["tool_arguments"]["search_attractions"]
            response = await activities_graph.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"{state['activities_task']}\n\n"
                        f"Use these validated attraction-search arguments: {json.dumps(args)}. "
                        "Do not invent tool inputs or claim mock data is live."
                    ),
                }]
            }, config={"recursion_limit": 15})
            return {"activities_result": str(response["messages"][-1].content)}

        async def accommodation(state: TripState) -> dict[str, str]:
            args = state["tool_arguments"]["search_hotels"]
            response = await accommodation_graph.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"{state['accommodation_task']}\n\n"
                        f"Use these validated hotel-search arguments: {json.dumps(args)}. "
                        "These hotels are mock suggestions, not confirmed availability."
                    ),
                }]
            }, config={"recursion_limit": 15})
            return {"accommodation_result": str(response["messages"][-1].content)}

        async def final_planner(state: TripState) -> dict[str, Any]:
            messages = [
                SystemMessage(content=(
                    "You are the WanderMind itinerary planner. Combine only the supplied research. "
                    "Preserve the user's constraints and validated arguments; don't assume number of nights "
                    "from the number of days. All travel data is illustrative, not verified live availability. "
                    "Do not invent hotel names, costs, attractions, exchange rates, or transport times. "
                    "If information is missing, describe it instead of inventing it."
                )),
                HumanMessage(content=(
                    f"Original request:\n{state['user_request']}\n\n"
                    f"Validated arguments:\n{json.dumps(state['tool_arguments'])}\n\n"
                    f"Activities research:\n{state['activities_result']}\n\n"
                    f"Accommodation research:\n{state['accommodation_result']}"
                )),
            ]
            if state.get("issues"):
                messages.append(HumanMessage(content=(
                    f"Previous plan:\n{state.get('final_plan', '')}\n\n"
                    f"Revise to address these validation issues:\n{json.dumps(state['issues'])}"
                )))
            response = await model.ainvoke(messages)
            return {
                "final_plan": str(response.content),
                "plan_attempts": state.get("plan_attempts", 0) + 1,
            }

        async def validate_final(state: TripState) -> dict[str, Any]:
            result = await validator_model.ainvoke([
                SystemMessage(content=(
                    "Review the itinerary against the original user request AND research evidence. "
                    "Flag unsupported or invented attractions, hotels, prices, availability, inaccurate "
                    "currencies, duration inconsistencies, and budget violations. "
                    "Don't confuse nightly hotel budget with total trip budget. "
                    "Return 'verified' only if no issues are found relative to the supplied mock data; "
                    "this does not mean hotels or prices are verified in the real world."
                )),
                HumanMessage(content=(
                    f"Request:\n{state['user_request']}\n\n"
                    f"Validated arguments:\n{json.dumps(state['tool_arguments'])}\n\n"
                    f"Activities research:\n{state['activities_result']}\n\n"
                    f"Accommodation research:\n{state['accommodation_result']}\n\n"
                    f"Final plan:\n{state['final_plan']}"
                )),
            ])
            return {"issues": result.issues, "status": result.status}

        def route_requirements(state: TripState) -> str:
            if not state.get("missing_details"):
                return "activities"
            if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
                return "needs_input"
            return "ask_user"

        def route_review(state: TripState) -> str:
            if state.get("status") == "verified" and not state.get("issues"):
                return "__end__"
            return "review_again" if state.get("plan_attempts", 0) < MAX_PLAN_ATTEMPTS else "__end__"

        def needs_input(_: TripState) -> dict[str, str]:
            return {"status": "needs_input"}

        workflow = StateGraph(TripState)
        workflow.add_node("coordinator", coordinator)
        workflow.add_node("requirements_check", requirements_check)
        workflow.add_node("ask_user", ask_user)
        workflow.add_node("needs_input", needs_input)
        workflow.add_node("activities", activities)
        workflow.add_node("accommodation", accommodation)
        workflow.add_node("final_planner", final_planner)
        workflow.add_node("validate_final", validate_final)
        workflow.add_edge(START, "coordinator")
        workflow.add_edge("coordinator", "requirements_check")
        workflow.add_conditional_edges("requirements_check", route_requirements, {
            "ask_user": "ask_user", "activities": "activities", "needs_input": "needs_input"
        })
        workflow.add_edge("ask_user", "coordinator")
        workflow.add_edge("needs_input", END)
        workflow.add_edge("activities", "accommodation")
        workflow.add_edge("accommodation", "final_planner")
        workflow.add_edge("final_planner", "validate_final")
        workflow.add_conditional_edges("validate_final", route_review, {
            "__end__": END, "review_again": "final_planner"
        })

        graph = workflow.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 50}
        if callbacks:
            config["callbacks"] = callbacks

        request = input("Enter your trip request (blank for Kyoto demo): ").strip()
        if not request:
            request = "Plan a 3-day Kyoto trip with a nightly hotel budget of $150."
        result = await graph.ainvoke({
            "user_request": request, "issues": [], "input_errors": [],
            "plan_attempts": 0, "clarification_rounds": 0, "status": "pending",
        }, config=config)

        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            answers: dict[str, dict[str, str]] = {}
            print("\nMissing or invalid details:")
            for item in payload["fields"]:
                print(f"- {item['tool']}.{item['field']}: {item['description']}")
                value = input("  Your answer: ").strip()
                answers.setdefault(item["tool"], {})[item["field"]] = value
            result = await graph.ainvoke(Command(resume=answers), config=config)

        print("\nStatus:", result.get("status", "unknown"))
        if result.get("status") == "needs_input":
            print("Could not complete required details:", json.dumps(result.get("missing_details", []), indent=2))
        else:
            print("Planning attempts:", result.get("plan_attempts", 0))
            print("Validation issues:", json.dumps(result.get("issues", []), indent=2))
            print("\nFinal itinerary (mock research):\n", result.get("final_plan", "No plan generated."))

    if langfuse_client is not None:
        langfuse_client.flush()


if __name__ == "__main__":
    asyncio.run(main())
