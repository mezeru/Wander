import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field, field_validator


class HotelStay(BaseModel):
    destination: str | None = None
    check_in: str | None = None
    check_out: str | None = None


class TripIntent(BaseModel):
    destinations: list[str] = Field(
        description=(
            "All destinations the user wants to visit. "
            "Extract place names regardless of whether "
            "they exist in the travel database."
        )
    )
    trip_days: int | None = None
    needs_activities: bool
    needs_accommodation: bool
    nightly_hotel_budget: float | None = None
    requires_itinerary: bool | None = None
    hotel_stays: list[HotelStay] = Field(default_factory=list)

    @field_validator("trip_days", mode="before")
    @classmethod
    def preserve_integer_days_only(cls, value: Any) -> int | None:
        return value if type(value) is int else None


class ValidationResult(BaseModel):
    issues: list[str] = Field(default_factory=list)
    status: Literal["verified", "needs_revision", "partial", "invalid"]


class ResearchWorkerState(TypedDict):
    job_id: str
    agent: Literal["activities", "accommodation"]
    destination: str
    task: str
    arguments: dict[str, Any]
    tool_name: NotRequired[str]
    user_request: NotRequired[str]


class TripState(TypedDict):
    user_request: str
    missing_details: NotRequired[list[dict[str, str]]]
    status_message: NotRequired[str]
    clarification_rounds: NotRequired[int]
    structured_plan: dict[str,Any] | None
    planner_error: NotRequired[str | None]
    plan_attempts: NotRequired[int]
    issues: NotRequired[list[str]]
    research_warnings: NotRequired[list[str]]
    status: NotRequired[
        Literal["pending", "needs_input", "verified", "invalid", "partial",
                "research_failed", "research_complete", "no_research", "workflow_error"]
    ]
    research_results: Annotated[list[dict[str, Any]], operator.add]
    trip_intent: dict[str, Any]
    research_jobs: NotRequired[list[dict[str, Any]]]
    validated_jobs: NotRequired[list[dict[str, Any]]]
    job_answers: NotRequired[dict[str, dict[str, Any]]]
    research_coverage: NotRequired[dict[str, Any]]
    itinerary_context: NotRequired[dict[str, Any]]

class ResearchJob(BaseModel):
    job_id: str
    agent: Literal["activities", "accommodation"]
    destination: str
    tool_name: str
    task: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class PlannedActivity(BaseModel):
    name: str
    evidence_job_id: str
    notes: str | None = None


class ItineraryDay(BaseModel):
    day_number: int
    destination: str
    activities: list[PlannedActivity] = Field(default_factory=list)
    accommodation: str | None = None
    accommodation_evidence_job_id: str | None = None
    transfer_notes: str | None = None


class TripItinerary(BaseModel):
    title: str
    days: list[ItineraryDay]
    warnings: list[str] = Field(default_factory=list)
