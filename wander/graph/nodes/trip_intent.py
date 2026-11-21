import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState
from ..validation import explicit_nightly_budget, parse_date
from .context import TripNodeContext


logger = logging.getLogger(__name__)
_WORD_DAYS = {
    word: count
    for count, word in enumerate(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        start=1,
    )
}
_NUMBER = r"(?:" + "|".join(_WORD_DAYS) + r"|\d+)"
_DAY_PATTERNS = (
    rf"(?<![\w.-])({_NUMBER})[- ]day\s+(?:[\w-]+\s+){{0,2}}?(?:trip|itinerary|vacation|holiday|visit)\b",
    rf"\bspend\s+({_NUMBER})\s+days?\s+(?:in|across|visiting|exploring)\b",
    rf"\b({_NUMBER})\s+days?\s+(?:in total|total|overall)\b",
    rf"\b(?:for|over)\s+({_NUMBER})\s+days?\b",
)
_DURATION_RANGE = re.compile(
    rf"\b{_NUMBER}(?:[- ]days?)?\s*(?:or|to|[-–])\s*{_NUMBER}[- ]days?\b",
    re.IGNORECASE,
)
_TRIP_REQUEST = re.compile(
    r"\b(?:plan|planning|organize|arrange)\b[^.!?\n]{0,70}\b(?:trip|vacation|holiday)\b"
    r"|\b(?:day[- ]by[- ]day|itinerary)\b|\b(?:sightseeing|road)\s+trip\b",
    re.IGNORECASE,
)
_LOOKUP = re.compile(
    r"^\s*(?:please\s+)?(?:find|show(?:\s+me)?|list|search\s+for|suggest|recommend)\s+(?:some\s+)?"
    r"(?P<target>hotels?|accommodations?|lodging|attractions?|sights|things to do)\b",
    re.IGNORECASE,
)
_EXPLICIT_DATES = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b"
)

_INTENT_PROMPT = """Extract travel requirements from the original request.
- destinations: Include every requested stop in travel order. Use the city name
  alone when a country is supplied only as context (Kyoto, Japan becomes Kyoto).
  Keep a regional qualifier when needed to identify a smaller city. A qualifier
  is not a separate stop. Return [] when no stop is named.
- requires_itinerary: True for a complete trip plan or day-by-day itinerary, even if
  the duration is missing. False for a lodging or attraction lookup. Use null if
  the request is too ambiguous to classify. An unrelated request is a lookup with
  both research flags false.
- trip_days: Extract only explicitly stated total itinerary days. Read 3-day,
  five-day, and for five days as durations in context. Do not use hotel nights,
  visit durations, or unrelated numbers. Use null when unspecified or ambiguous.
- needs_activities: True for sightseeing, attractions, or a general trip plan,
  unless the user excludes activities. False for a hotel-only lookup.
- needs_accommodation: True when lodging is requested. For a general multi-day trip
  both categories are normally relevant, unless the user limits the scope.
  False for attractions-only requests.
- nightly_hotel_budget: Extract an explicitly nightly room budget only. A total
  trip budget is insufficient. Leave it null when missing.
- hotel_stays: Include only explicitly supplied check-in or check-out dates tied to
  the appropriate city. Use ISO YYYY-MM-DD when the year is given. Leave missing
  dates null. Do not treat generic trip dates as lodging dates or infer a year.
  Leave the list empty if there are no explicit lodging dates.

Examples:
"Plan a 3-day Kyoto trip. Include sightseeing and rooms at $150 per night."
  Kyoto, 3 days, itinerary true, both research flags true, nightly budget 150.
"Find hotels in Tokyo." Tokyo, accommodation only, itinerary false, duration null.
"Show me attractions in Paris." Paris, activities only, itinerary false, duration null.
"Plan a trip to Osaka with a total budget of $500."
  Osaka, itinerary true, duration null, nightly budget null.
"Plan a two-day trip to Chickasha, Oklahoma."
  One destination, Chickasha, Oklahoma, two days.
"""


def _explicit_duration(request: str) -> tuple[int | None, bool]:
    exact: set[int] = set()
    for index, pattern in enumerate(_DAY_PATTERNS):
        for match in re.finditer(pattern, request, re.IGNORECASE):
            if index == len(_DAY_PATTERNS) - 1:
                # A generic "for N days" must describe the trip itself.
                clause = re.split(r"[.!?;\n]", request[:match.start()])[-1]
                if not re.search(r"\b(?:plan|trip|visit|travel|vacation|holiday|itinerary)\b", clause, re.I):
                    continue
                if re.search(r"\b(?:hotel|accommodation|tour|pass|ticket|car rental)\b", clause, re.I):
                    continue
            amount = match[1].casefold()
            exact.add(int(amount) if amount.isdigit() else _WORD_DAYS[amount])
    ambiguous = len(exact) > 1 or _DURATION_RANGE.search(request) is not None
    return (next(iter(exact)) if len(exact) == 1 and not ambiguous else None), ambiguous


def _explicit_dates(request: str) -> set[str]:
    dates = set()
    for match in _EXPLICIT_DATES.finditer(request):
        try:
            dates.add(parse_date(match[0]))
        except ValueError:
            continue
    return dates


def create_extract_trip_intent_node(context: TripNodeContext):
    async def extract_trip_intent(state: TripState) -> dict[str, Any]:
        request = state["user_request"]
        result = await context.intent_model.ainvoke(
            [SystemMessage(content=_INTENT_PROMPT), HumanMessage(content=request)]
        )
        intent = result.model_dump()
        logger.debug("Raw extracted intent: %s", intent)
        destinations: dict[str, str] = {}
        for destination in intent["destinations"]:
            if destination.strip():
                destinations.setdefault(destination.strip().casefold(), destination.strip())
        intent["destinations"] = list(destinations.values())

        recovered_days, ambiguous = _explicit_duration(request)
        lookup = _LOOKUP.match(request)
        itinerary_language = _TRIP_REQUEST.search(request) is not None
        separate_plan_request = re.search(
            r"(?:\band\b|[.;])\s*(?:also\s+)?(?:plan|build|create)\b", request, re.I,
        )
        if lookup is not None and not (itinerary_language and separate_plan_request):
            asks_for_hotels = lookup["target"].casefold().startswith(("hotel", "accommodation", "lodging"))
            other_type = (
                r"\b(?:attractions?|sightseeing|things to do)\b" if asks_for_hotels
                else r"\b(?:hotels?|accommodations?|lodging)\b"
            )
            if not re.search(other_type, request, re.IGNORECASE):
                intent.update(needs_activities=not asks_for_hotels, needs_accommodation=asks_for_hotels)
            intent["requires_itinerary"] = False
        elif itinerary_language:
            intent["requires_itinerary"] = True
            intent["needs_activities"] = not bool(re.search(
                r"\b(?:no|without|skip|exclude)\s+(?:any\s+)?(?:sightseeing|activities|attractions)\b",
                request, re.I,
            ))
            lodging_language = r"\b(?:hotels?|accommodations?|lodging)\b"
            if re.search(r"\b(?:no|without|skip|exclude)\s+(?:any\s+)?" + lodging_language, request, re.I):
                intent["needs_accommodation"] = False
            elif re.search(lodging_language, request, re.I):
                intent["needs_accommodation"] = True
            elif re.search(r"\bsightseeing\s+(?:trip|itinerary)\b", request, re.I):
                intent["needs_accommodation"] = False
        elif intent["requires_itinerary"] is None and (intent["needs_activities"] or intent["needs_accommodation"]):
            intent["requires_itinerary"] = bool(intent["needs_activities"] and (
                intent["trip_days"] is not None or recovered_days is not None or ambiguous
            ))

        if ambiguous or intent["requires_itinerary"] is False:
            intent["trip_days"] = None
        elif recovered_days is not None:
            intent["trip_days"] = recovered_days
        elif not re.search(r"\bdays?\b", request, re.IGNORECASE):
            intent["trip_days"] = None

        intent["nightly_hotel_budget"] = (
            explicit_nightly_budget(request) if intent["needs_accommodation"] else None
        )
        dates = _explicit_dates(request) if intent["hotel_stays"] else set()
        for stay in intent["hotel_stays"]:
            for key in ("check_in", "check_out"):
                try:
                    normalized = parse_date(stay[key])
                except ValueError:
                    normalized = None
                stay[key] = normalized if normalized in dates else None
        intent["hotel_stays"] = [
            stay for stay in intent["hotel_stays"] if stay["check_in"] or stay["check_out"]
        ]

        logger.info(
            "Extracted intent: destinations=%d days=%s itinerary=%s activities=%s accommodation=%s nightly_budget_supplied=%s",
            len(intent["destinations"]), intent["trip_days"], intent["requires_itinerary"],
            intent["needs_activities"], intent["needs_accommodation"],
            intent["nightly_hotel_budget"] is not None,
        )
        return {"trip_intent": intent}

    return extract_trip_intent
