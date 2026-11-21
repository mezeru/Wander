from typing import Any

from pydantic import BaseModel, Field


class Price(BaseModel):
    amount: float = Field(ge=0)
    currency: str


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class Attraction(BaseModel):
    id: str
    name: str
    category: str
    estimated_duration_hours: float | None = Field(default=None, gt=0)
    price: Price | None = None
    source: str
    verified_live: bool = False
    destination: str | None = None
    provider: str | None = None
    coordinates: Coordinates | None = None
    description: str | None = None
    source_url: str | None = None
    address: dict[str, Any] | str | None = None
    website: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class AttractionSearchResult(BaseModel):
    destination: str
    attractions: list[Attraction] = Field(default_factory=list)


class DestinationPlan(BaseModel):
    destination: str
    attractions: list[Attraction] = Field(default_factory=list)


class Hotel(BaseModel):
    id: str
    name: str
    nightly_price: Price | None = None
    availability_verified: bool = False
    source: str
    destination: str | None = None
    provider: str | None = None
    category: str | None = None
    address: str | None = None
    coordinates: Coordinates | None = None
    source_url: str | None = None
    website: str | None = None
    pricing_available: bool = False
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class HotelSearchResult(BaseModel):
    destination: str
    hotels: list[Hotel] = Field(default_factory=list)