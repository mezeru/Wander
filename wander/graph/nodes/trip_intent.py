from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState
from .context import TripNodeContext


def create_extract_trip_intent_node(context: TripNodeContext):
    async def extract_trip_intent(state: TripState) -> dict[str, Any]:
        result = await context.intent_model.ainvoke(
            [
                SystemMessage(
                    content="""
            You extract travel requirements from user requests.

            Return every destination explicitly mentioned.
            Never restrict destinations to a known dataset.

            For complete trip planning, both activities
            and accommodation research are normally needed.

            Never invent missing dates or budgets.

            Examples:
            "Plan a 3-day Kyoto trip"
            -> destinations: ["Kyoto"]

            "Visit Tokyo, Kyoto and Osaka"
            -> destinations: ["Tokyo", "Kyoto", "Osaka"]

            "Find hotels in Ljubljana"
            -> destinations: ["Ljubljana"]
            """
                ),
                HumanMessage(content=state["user_request"]),
            ]
        )
        return {"trip_intent": result.model_dump()}

    return extract_trip_intent
