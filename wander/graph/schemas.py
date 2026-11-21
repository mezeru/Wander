import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class ResearchTasks(BaseModel):
    activities_task: str | None = Field(
        default=None, description="Instructions for the Activities Agent"
    )
    accommodation_task: str | None = Field(
        default=None, description="Instructions for the Accommodation Agent"
    )
    tool_arguments: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Arguments grouped by MCP tool name; omit unknown values.",
    )
    destination: str | None = Field(
        default=None,
        description=(
            "The explicitly specified travel destination. "
            "For example, 'Plan a 3-day Kyoto trip' means "
            "destination='Kyoto'. Must not be null if one "
            "unambiguous destination appears in the request."
        ),
    )


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


class ValidationResult(BaseModel):
    issues: list[str] = Field(default_factory=list)
    status: Literal["verified", "needs_revision", "partial", "invalid"]


class ResearchWorkerState(TypedDict):
    job_id: str
    agent: Literal["activities", "accommodation"]
    destination: str
    task: str
    arguments: dict[str, Any]


class TripState(TypedDict):
    user_request: str
    activities_task: NotRequired[str]
    accommodation_task: NotRequired[str]
    tool_arguments: NotRequired[dict[str, dict[str, Any]]]
    user_context: NotRequired[dict[str, dict[str, Any]]]
    missing_details: NotRequired[list[dict[str, str]]]
    input_errors: NotRequired[list[dict[str, str]]]
    clarification_rounds: NotRequired[int]
    structured_plan: dict[str,Any] | None
    plan_attempts: NotRequired[int]
    issues: NotRequired[list[str]]
    research_warnings: NotRequired[list[str]]
    status: NotRequired[
        Literal["pending", "needs_input", "verified", "needs_revision", "invalid", "ready_for_dispatch", "partial", "research_failed"]
    ]
    research_results: Annotated[list[dict[str, Any]], operator.add]
    trip_intent: TripIntent
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