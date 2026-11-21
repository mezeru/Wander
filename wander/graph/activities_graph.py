from pydantic import BaseModel, Field

from .mcp_schemas import Coordinates, Price
from .specialist import build_specialist_graph


class AttractionFinding(BaseModel):
    name: str
    id: str | None = None
    category: str | None = None
    location: str | None = None
    description: str | None = None
    coordinates: Coordinates | None = None
    provider: str | None = None
    relevance: str | None = None
    price: Price | None = None
    estimated_duration_hours: float | None = None


class ActivitiesReport(BaseModel):
    attractions: list[AttractionFinding]
    limitations: list[str] = Field(default_factory=list)


_PROMPT = """Prioritize the numbered attraction candidates for this destination
using the user's stated interests. Return numbers in suggested order. A note is
optional and may briefly explain relevance using only supplied facts. The caller
carries through every candidate's actual name, provider ID, and source fields.
Return brief limitations only when supported. Leave unknown opening hours,
admission prices, and visit durations unknown. Do not add outside facts or
invent attractions to fill days.
"""


async def build_activities_graph(tools: list, model):
    return build_specialist_graph("activities", tools, model, ActivitiesReport, "attractions", _PROMPT)
