import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState
from .context import TripNodeContext


_DAY_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_DAY_TOKEN = (
    r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
)


def _explicit_trip_days(request: str) -> int | None:
    patterns = (
        rf"\b(?P<days>{_DAY_TOKEN})\s*-\s*days?\s+trip\b",
        rf"\bfor\s+(?P<days>{_DAY_TOKEN})\s+days?\b",
        rf"\b(?P<days>{_DAY_TOKEN})\s+days?\s+in\s+[A-Za-z]",
    )
    matches = [
        match
        for pattern in patterns
        for match in re.finditer(pattern, request, re.IGNORECASE)
    ]
    if len(matches) != 1:
        return None

    value = matches[0].group("days").casefold()
    return _DAY_COUNTS.get(value, int(value) if value.isdigit() else None)


def create_extract_trip_intent_node(context: TripNodeContext):
    async def extract_trip_intent(state: TripState) -> dict[str, Any]:
        result = await context.intent_model.ainvoke(
            [
                SystemMessage(
                    content="""
            Extract every TripIntent field from the user's request:
            - destinations: all explicitly named places, in travel order; do not
              restrict them to a known dataset.
            - trip_days: an explicitly stated total number of days, or null when
              not stated. Never invent a duration.
            - needs_activities: true for sightseeing or general trip planning;
              general trip planning usually needs activities research.
            - needs_accommodation: true when lodging is requested or when a
              complete trip plan is requested. For a hotel-only request, set
              needs_activities=false. For an activities-only request, set
              needs_accommodation=false.
            - nightly_hotel_budget: only an explicitly stated per-night hotel
              budget; a total trip budget is not a nightly hotel budget.
            Never invent dates or budgets.

            Example:
            User: "Plan a 3-day trip to Kyoto, Japan. Include sightseeing
            attractions and accommodation. My nightly hotel budget is $150."
            -> {
              "destinations": ["Kyoto"],
              "trip_days": 3,
              "needs_activities": true,
              "needs_accommodation": true,
              "nightly_hotel_budget": 150.0
            }

            "Visit Tokyo, Kyoto and Osaka"
            -> destinations: ["Tokyo", "Kyoto", "Osaka"], trip_days: null

            "Find hotels in Ljubljana"
            -> destinations: ["Ljubljana"], trip_days: null,
               needs_activities: false, needs_accommodation: true,
               nightly_hotel_budget: null

            "Plan a two-day trip to Chickasha with a budget of 300"
            -> destinations: ["Chickasha"], trip_days: 2,
               needs_activities: true, needs_accommodation: true,
               nightly_hotel_budget: null
            """
                ),
                HumanMessage(content=state["user_request"]),
            ]
        )
        trip_intent = result.model_dump()
        if trip_intent.get("trip_days") is None:
            explicit_days = _explicit_trip_days(state["user_request"])
            if explicit_days is not None:
                trip_intent["trip_days"] = explicit_days
        return {"trip_intent": trip_intent}

    return extract_trip_intent
