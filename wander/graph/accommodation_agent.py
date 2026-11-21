from pydantic import BaseModel, Field

from .mcp_schemas import Coordinates, Price
from .specialist import build_specialist_graph


class HotelFinding(BaseModel):
    name: str
    id: str | None = None
    location: str | None = None
    coordinates: Coordinates | None = None
    provider: str | None = None
    nightly_price: Price | None = None
    nightly_rate: float | None = None
    currency: str | None = None
    availability_verified: bool = False


class AccommodationReport(BaseModel):
    hotels: list[HotelFinding]
    limitations: list[str] = Field(default_factory=list)


_PROMPT = """Prioritize the numbered lodging candidates for the assigned
destination. Return candidate numbers once in suggested order and omit per-place
notes. The caller preserves property names, IDs, explicit English labels, and
source fields. The budget and dates are constraints. Geoapify discovers places.
A missing rate leaves affordability unknown. Give brief source limitations.
Never imply a booking, availability, or budget match without evidence. Do not
invent amenities, ratings, translations, or prices.
"""


async def build_accommodation_graph(tools: list, model):
    return build_specialist_graph("accommodation", tools, model, AccommodationReport, "hotels", _PROMPT)
