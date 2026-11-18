import re
from datetime import date, datetime
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError


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
    for fmt in (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
    ):
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
            match = re.fullmatch(
                r"\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\$?",
                text,
            )
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
            values = [
                part.strip()
                for part in re.split(r"[,;]|\band\b", value)
                if part.strip()
            ]
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


def clean_task(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.lower() in {"null", "none", ""}:
        return None
    return cleaned


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
                    {
                        "type": "object",
                        "properties": {field: properties[field]},
                        "$defs": schema.get("$defs", {}),
                    },
                    format_checker=FormatChecker(),
                ).validate({field: converted})
                valid[field] = converted
            except (ValueError, TypeError, ValidationError) as exc:
                detail = exc.message if isinstance(exc, ValidationError) else str(exc)
                issues.append(issue(tool_name, field, f"Invalid {field}: {detail}"))
        for field in schema.get("required", []):
            if field not in valid:
                description = (
                    properties.get(field, {}).get("description")
                    or f"Please provide {field}."
                )
                issues.append(issue(tool_name, field, description))
        normalized[tool_name] = valid

    hotel = normalized.get("search_hotels", {})
    start, end = hotel.get("check_in"), hotel.get("check_out")
    if start and end:
        nights = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if nights <= 0:
            issues.append(
                issue(
                    "search_hotels",
                    "check_out",
                    "Check-out must be later than check-in.",
                )
            )
        else:
            match = re.search(r"\b(\d+)\s*[- ]?days?\b", user_request, re.IGNORECASE)
            if (
                match
                and nights > int(match.group(1))
                and not context.get("__trip__", {}).get("confirm_extra_nights")
            ):
                issues.append(
                    issue(
                        "__trip__",
                        "confirm_extra_nights",
                        f"The stay has {nights} hotel nights, but the trip says "
                        f"{match.group(1)} days. Accept the extra nights? (yes/no)",
                    )
                )

    return normalized, dedupe_issues(issues)
