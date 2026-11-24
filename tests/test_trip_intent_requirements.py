import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from wander.graph.nodes.context import TripNodeContext
from wander.graph.nodes.job_creation import create_research_jobs_node
from wander.graph.nodes.needs_input import needs_input
from wander.graph.nodes.requirements_check import create_requirements_check_node
from wander.graph.nodes.routes import route_requirements
from wander.graph.nodes.trip_intent import create_extract_trip_intent_node
from wander.graph.schemas import (
    ItineraryDay,
    ResearchTasks,
    TripIntent,
    TripItinerary,
)
from wander.graph.trip_coordinator import _choose_trip_request, build_workflow


ACTIVITIES_SCHEMA = {
    "type": "object",
    "properties": {"destination": {"type": "string"}},
    "required": ["destination"],
}
HOTEL_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "check_in": {"type": "string", "format": "date"},
        "check_out": {"type": "string", "format": "date"},
        "budget": {"type": "number"},
    },
    "required": ["destination", "check_in", "check_out", "budget"],
}


class FakeIntentModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


class FakeCoordinatorModel:
    async def ainvoke(self, _messages):
        return ResearchTasks(
            activities_task="Research sightseeing attractions",
            accommodation_task="Research accommodation",
        )


class FakeStructuredModel:
    def __init__(self, destinations):
        self.destinations = destinations

    async def ainvoke(self, _messages):
        return TripItinerary(
            title="Test itinerary",
            days=[
                ItineraryDay(day_number=index, destination=destination)
                for index, destination in enumerate(self.destinations, start=1)
            ],
        )


class FakePlannerModel:
    def __init__(self, destinations):
        self.structured_model = FakeStructuredModel(destinations)

    def with_structured_output(self, _schema, method):
        assert method == "json_schema"
        return self.structured_model


class FakeResearchGraph:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, state):
        self.calls.append(state)
        return {"messages": [AIMessage(content="Research completed.")]}


def _context(intent, itinerary_destinations):
    activities_graph = FakeResearchGraph()
    accommodation_graph = FakeResearchGraph()
    context = TripNodeContext(
        model=FakePlannerModel(itinerary_destinations),
        coordinator_model=FakeCoordinatorModel(),
        intent_model=FakeIntentModel(intent),
        schemas={
            "search_attractions": ACTIVITIES_SCHEMA,
            "search_hotels": HOTEL_SCHEMA,
        },
        tool_specs=[],
        activities_graph=activities_graph,
        accommodation_graph=accommodation_graph,
    )
    return context, activities_graph, accommodation_graph


def _initial_state(request, job_answers=None):
    return {
        "user_request": request,
        "issues": [],
        "input_errors": [],
        "job_answers": job_answers or {},
        "plan_attempts": 0,
        "clarification_rounds": 0,
        "status": "pending",
        "research_results": [],
    }


@pytest.mark.parametrize(
    ("user_request", "days"),
    [
        ("Plan a 3-day trip to Kyoto.", 3),
        ("Plan a three-day trip to Kyoto.", 3),
        ("Plan for 5 days.", 5),
        ("Plan two days in Kyoto.", 2),
        ("Plan two days in Kyoto and two days in Tokyo.", None),
    ],
)
def test_explicit_duration_fallback_only_fills_missing_model_value(user_request, days):
    model = FakeIntentModel(TripIntent(
        destinations=["Kyoto"],
        trip_days=None,
        needs_activities=True,
        needs_accommodation=False,
    ))
    node = create_extract_trip_intent_node(TripNodeContext(intent_model=model))

    result = asyncio.run(node({"user_request": user_request}))

    assert result["trip_intent"]["trip_days"] == days


def test_general_trip_prompt_distinguishes_research_scope_and_budget():
    model = FakeIntentModel(TripIntent(
        destinations=["Chickasha, Oklahoma"],
        trip_days=None,
        needs_activities=True,
        needs_accommodation=True,
        nightly_hotel_budget=None,
    ))
    node = create_extract_trip_intent_node(TripNodeContext(intent_model=model))
    result = asyncio.run(node({
        "user_request": "Plan a two-day trip to Chickasha, Oklahoma with a budget of 300."
    }))

    prompt = model.messages[0].content
    assert result["trip_intent"]["trip_days"] == 2
    assert result["trip_intent"]["destinations"] == ["Chickasha, Oklahoma"]
    assert result["trip_intent"]["needs_activities"] is True
    assert result["trip_intent"]["nightly_hotel_budget"] is None
    assert '"trip_days": 3' in prompt
    assert '"needs_activities": true' in prompt
    assert '"needs_accommodation": true' in prompt
    assert '"nightly_hotel_budget": 150.0' in prompt
    assert "general trip planning usually needs activities research" in prompt
    assert "a total trip budget is not a nightly hotel budget" in prompt
    jobs = asyncio.run(create_research_jobs_node(TripNodeContext())({
        "trip_intent": result["trip_intent"],
    }))
    assert {job["agent"] for job in jobs["research_jobs"]} == {
        "activities", "accommodation",
    }


@pytest.mark.parametrize(
    ("intent", "answer", "expected_field", "destinations"),
    [
        (
            TripIntent(
                destinations=[],
                trip_days=3,
                needs_activities=True,
                needs_accommodation=False,
            ),
            {"destination": "Chickasha"},
            "destination",
            ["Chickasha", "Chickasha", "Chickasha"],
        ),
        (
            TripIntent(
                destinations=["Chickasha"],
                trip_days=2,
                needs_activities=False,
                needs_accommodation=False,
            ),
            {"research_scope": "activities"},
            "research_scope",
            ["Chickasha", "Chickasha"],
        ),
    ],
)
def test_empty_job_clarification_updates_intent_and_creates_jobs(
    intent, answer, expected_field, destinations
):
    context, activities_graph, _ = _context(intent, destinations)
    graph = build_workflow(context)
    config = {"configurable": {"thread_id": f"empty-job-{expected_field}"}}

    paused = asyncio.run(graph.ainvoke(
        _initial_state("Plan a trip."),
        config=config,
    ))
    fields = paused["__interrupt__"][0].value["fields"]
    assert expected_field in {item["field"] for item in fields}
    assert activities_graph.calls == []

    result = asyncio.run(graph.ainvoke(
        Command(resume={"__trip__": answer}),
        config=config,
    ))

    assert result["status"] == "verified"
    assert result["trip_intent"]["needs_activities"] is True
    assert len(result["research_jobs"]) == 1
    assert len(result["research_results"]) == 1
    assert len(activities_graph.calls) == 1


def test_missing_duration_and_hotel_fields_resume_once_before_dispatch():
    intent = TripIntent(
        destinations=["Kyoto"],
        trip_days=None,
        needs_activities=True,
        needs_accommodation=True,
        nightly_hotel_budget=150,
    )
    context, activities_graph, accommodation_graph = _context(
        intent, ["Kyoto", "Kyoto", "Kyoto"]
    )
    graph = build_workflow(context)
    config = {"configurable": {"thread_id": "global-and-hotel-clarification"}}

    paused = asyncio.run(graph.ainvoke(
        _initial_state("Plan sightseeing and hotels in Kyoto."),
        config=config,
    ))
    fields = paused["__interrupt__"][0].value["fields"]
    assert {(item["job_id"], item["field"]) for item in fields} == {
        ("__trip__", "trip_days"),
        ("accommodation-0", "check_in"),
        ("accommodation-0", "check_out"),
    }
    assert activities_graph.calls == []
    assert accommodation_graph.calls == []

    result = asyncio.run(graph.ainvoke(
        Command(resume={
            "__trip__": {"trip_days": "3"},
            "accommodation-0": {
                "check_in": "2027-04-10",
                "check_out": "2027-04-13",
            },
        }),
        config=config,
    ))

    assert result["status"] == "verified"
    assert result["trip_intent"]["trip_days"] == 3
    assert result["trip_intent"]["nightly_hotel_budget"] == 150
    assert result["trip_intent"]["needs_activities"] is True
    assert result["trip_intent"]["needs_accommodation"] is True
    assert len(result["research_jobs"]) == 2
    assert len(result["research_results"]) == 2
    assert len(activities_graph.calls) == 1
    assert len(accommodation_graph.calls) == 1
    hotel_arguments = result["validated_jobs"][1]["arguments"]
    assert hotel_arguments == {
        "destination": "Kyoto",
        "budget": 150.0,
        "check_in": "2027-04-10",
        "check_out": "2027-04-13",
    }
    assert "2027-04-10" in accommodation_graph.calls[0]["messages"][0]["content"]


def test_hotel_only_request_does_not_require_itinerary_duration():
    intent = TripIntent(
        destinations=["Tokyo"],
        trip_days=None,
        needs_activities=False,
        needs_accommodation=True,
    )
    context, activities_graph, accommodation_graph = _context(intent, [])
    graph = build_workflow(context)
    config = {"configurable": {"thread_id": "hotel-only-no-trip-days"}}

    paused = asyncio.run(graph.ainvoke(
        _initial_state("Find hotels in Tokyo."),
        config=config,
    ))
    fields = paused["__interrupt__"][0].value["fields"]
    assert {item["field"] for item in fields} == {
        "check_in", "check_out", "budget",
    }
    assert "trip_days" not in {item["field"] for item in fields}

    result = asyncio.run(graph.ainvoke(
        Command(resume={
            "accommodation-0": {
                "check_in": "2027-04-10",
                "check_out": "2027-04-12",
                "budget": "150",
            }
        }),
        config=config,
    ))

    assert result["status"] == "verified"
    assert result.get("structured_plan") is None
    assert activities_graph.calls == []
    assert len(accommodation_graph.calls) == 1
    assert "2027-04-10" in accommodation_graph.calls[0]["messages"][0]["content"]


def test_hotel_only_partial_research_is_not_reported_as_fully_verified():
    assert needs_input({
        "trip_intent": {"needs_activities": False},
        "research_coverage": {
            "completed": 1,
            "failed": ["accommodation-1"],
            "no_data": [],
            "missing": [],
            "duplicate": [],
            "unknown": [],
        },
    }) == {"status": "partial"}


def test_multi_destination_dispatch_fans_out_four_jobs_and_fans_in():
    intent = TripIntent(
        destinations=["Kyoto", "Tokyo"],
        trip_days=5,
        needs_activities=True,
        needs_accommodation=True,
        nightly_hotel_budget=150,
    )
    itinerary_destinations = ["Kyoto", "Kyoto", "Kyoto", "Tokyo", "Tokyo"]
    context, activities_graph, accommodation_graph = _context(
        intent, itinerary_destinations
    )
    graph = build_workflow(context)
    job_answers = {
        "accommodation-0": {
            "check_in": "2027-04-10",
            "check_out": "2027-04-13",
        },
        "accommodation-1": {
            "check_in": "2027-04-13",
            "check_out": "2027-04-15",
        },
    }

    result = asyncio.run(graph.ainvoke(
        _initial_state(
            "Plan a 5-day trip to Kyoto and Tokyo. Include attractions and accommodation.",
            job_answers,
        ),
        config={"configurable": {"thread_id": "multi-destination-four-jobs"}},
    ))

    assert [job["job_id"] for job in result["research_jobs"]] == [
        "activities-0",
        "accommodation-0",
        "activities-1",
        "accommodation-1",
    ]
    assert len(result["validated_jobs"]) == 4
    assert len(result["research_results"]) == 4
    assert len(activities_graph.calls) == 2
    assert len(accommodation_graph.calls) == 2
    assert [day["destination"] for day in result["structured_plan"]["days"]] == (
        itinerary_destinations
    )
    assert result["trip_intent"]["destinations"] == ["Kyoto", "Tokyo"]


def test_zero_jobs_produces_actionable_scope_clarification():
    node = create_requirements_check_node(SimpleNamespace(schemas={}))
    state = {
        "research_jobs": [],
        "job_answers": {},
        "trip_intent": {
            "destinations": ["Chickasha"],
            "trip_days": 2,
            "needs_activities": False,
            "needs_accommodation": False,
        },
        "clarification_rounds": 0,
    }

    result = node(state)

    assert result["missing_details"][0]["field"] == "research_scope"
    assert "activities" in result["missing_details"][0]["description"]
    assert route_requirements({
        **state,
        "missing_details": result["missing_details"],
    }) == "ask_user"


def test_cli_default_is_explicit_and_nonblank_request_is_preserved(capsys):
    default_request = _choose_trip_request(" \t ")
    assert default_request.startswith("Plan a 3-day trip to Kyoto, Japan.")
    assert "sightseeing attractions and accommodation" in default_request
    assert "nightly hotel budget is $150" in default_request
    assert capsys.readouterr().out.strip() == "Using the default Kyoto demo request."

    request = "  Plan a custom trip to Kyoto.  "
    assert _choose_trip_request(request) == request
    assert capsys.readouterr().out == ""
