import asyncio
import json
from types import SimpleNamespace

import pytest
import wander.graph.nodes as nodes_module
from wander.graph.nodes.itinerary_context import (
    allocate_days,
    group_research_by_destination,
    prepare_itinerary_context,
)
from wander.graph.nodes.context import TripNodeContext
from wander.graph.schemas import TripIntent, TripItinerary
from wander.graph.trip_coordinator import build_workflow


def test_allocate_days_distributes_remainder_in_destination_order():
    assert allocate_days(["Tokyo", "Kyoto", "Osaka"], 7) == {
        "Tokyo": 3,
        "Kyoto": 2,
        "Osaka": 2,
    }
    assert allocate_days(["Tokyo", "Kyoto"], 5) == {
        "Tokyo": 3,
        "Kyoto": 2,
    }


def test_allocate_days_single_destination_gets_all_days():
    assert allocate_days(["Tokyo"], 4) == {"Tokyo": 4}


@pytest.mark.parametrize(
    ("destinations", "days", "message"),
    [
        ([], 1, "At least one destination"),
        (["Tokyo"], 0, "positive integer"),
        (["Tokyo"], -1, "positive integer"),
        (["Tokyo", "Kyoto"], 1, "Not enough days"),
        (["Tokyo", "tokyo"], 2, "Duplicate destinations"),
        (["Tokyo", " Tokyo "], 2, "Duplicate destinations"),
        (["Tokyo", ""], 2, "non-empty strings"),
        (["Tokyo", None], 2, "non-empty strings"),
        (["Tokyo"], True, "positive integer"),
    ],
)
def test_allocate_days_rejects_invalid_inputs(destinations, days, message):
    with pytest.raises(ValueError, match=message):
        allocate_days(destinations, days)


def _job(job_id, agent, destination):
    return {
        "job_id": job_id,
        "agent": agent,
        "destination": destination,
    }


def _result(job_id, agent, destination, status="completed", content="evidence"):
    return {
        "job_id": job_id,
        "agent": agent,
        "destination": destination,
        "status": status,
        "result": content,
        "metadata": {"source": job_id},
    }


def _group_state(results):
    return {
        "trip_intent": {"destinations": ["Tokyo", "Kyoto", "Osaka"]},
        "validated_jobs": [
            _job("activities-0", "activities", "Tokyo"),
            _job("activities-1", "activities", "Kyoto"),
            _job("activities-2", "activities", "Osaka"),
            _job("accommodation-0", "accommodation", "Tokyo"),
        ],
        "research_results": results,
    }


def test_research_grouping_preserves_destination_and_assignment_order():
    results = [
        _result("activities-2", "activities", "Osaka", content="Osaka evidence"),
        _result("activities-0", "activities", "Tokyo", content="Tokyo evidence"),
        _result("activities-1", "activities", "Kyoto", content="Kyoto evidence"),
    ]
    grouped = group_research_by_destination(_group_state(results))

    assert list(grouped) == ["Tokyo", "Kyoto", "Osaka"]
    assert grouped["Tokyo"]["activities"] == [results[1]]
    assert grouped["Kyoto"]["activities"] == [results[2]]
    assert grouped["Osaka"]["activities"] == [results[0]]
    assert grouped["Tokyo"]["accommodation"] == []


def test_research_grouping_keeps_multiple_results_in_assignment_order():
    state = {
        "trip_intent": {"destinations": ["Tokyo"]},
        "validated_jobs": [
            _job("activities-a", "activities", "Tokyo"),
            _job("activities-b", "activities", "Tokyo"),
        ],
        "research_results": [
            _result("activities-b", "activities", "Tokyo", content="second"),
            _result("activities-a", "activities", "Tokyo", content="first"),
        ],
    }

    evidence = group_research_by_destination(state)["Tokyo"]["activities"]
    assert [item["job_id"] for item in evidence] == [
        "activities-a",
        "activities-b",
    ]


def test_research_grouping_excludes_failed_and_no_data_results():
    grouped = group_research_by_destination(
        _group_state([
            _result("activities-0", "activities", "Tokyo"),
            _result(
                "activities-1", "activities", "Kyoto",
                status="failed", content=None,
            ),
            _result(
                "activities-2", "activities", "Osaka",
                status="no_data", content=None,
            ),
        ])
    )
    assert len(grouped["Tokyo"]["activities"]) == 1
    assert grouped["Kyoto"]["activities"] == []
    assert grouped["Osaka"]["activities"] == []


def test_research_grouping_rejects_mismatches_unknowns_and_duplicate_ids():
    state = _group_state([
        _result("activities-0", "activities", "Kyoto"),
        _result("activities-1", "accommodation", "Kyoto"),
        _result("activities-2", "activities", "Osaka"),
        _result("activities-2", "activities", "Osaka", content="duplicate"),
        _result("unknown-1", "activities", "Tokyo"),
        {"agent": "activities", "destination": "Tokyo", "status": "completed"},
    ])
    grouped = group_research_by_destination(state)
    assert grouped["Tokyo"]["activities"] == []
    assert grouped["Kyoto"]["activities"] == []
    assert grouped["Osaka"]["activities"] == []

    prepared = prepare_itinerary_context({
        **state,
        "trip_intent": {
            "destinations": ["Tokyo", "Kyoto", "Osaka"],
            "trip_days": 7,
        },
        "research_coverage": {},
    })["itinerary_context"]
    issues = prepared["research_coverage"]["evidence_issues"]
    assert {issue["reason"] for issue in issues} >= {
        "assignment_mismatch",
        "duplicate_result_job_id",
        "unknown_result_job_id",
        "missing_result_job_id",
    }


def test_research_grouping_rejects_duplicate_authoritative_job_ids():
    state = {
        "trip_intent": {"destinations": ["Tokyo"]},
        "validated_jobs": [
            _job("activities-0", "activities", "Tokyo"),
            _job("activities-0", "activities", "Tokyo"),
        ],
        "research_results": [
            _result("activities-0", "activities", "Tokyo"),
        ],
    }
    assert group_research_by_destination(state)["Tokyo"]["activities"] == []


def test_prepare_context_keeps_partial_destination_category_coverage():
    state = {
        "trip_intent": {
            "destinations": ["Tokyo", "Kyoto", "Osaka"],
            "trip_days": 7,
        },
        "validated_jobs": [
            _job("activities-0", "activities", "Tokyo"),
            _job("accommodation-0", "accommodation", "Tokyo"),
            _job("activities-1", "activities", "Kyoto"),
        ],
        "research_results": [
            _result("activities-0", "activities", "Tokyo"),
            _result("activities-1", "activities", "Kyoto"),
        ],
        "research_coverage": {
            "expected": 3,
            "completed": 2,
            "missing": ["accommodation-0"],
        },
    }
    context = prepare_itinerary_context(state)["itinerary_context"]
    assert context["destinations"] == ["Tokyo", "Kyoto", "Osaka"]
    assert context["total_days"] == 7
    assert context["research_coverage"]["missing"] == ["accommodation-0"]
    assert context["research_coverage"]["by_destination"]["Tokyo"]["accommodation"] == {
        "expected": 1,
        "usable": 0,
        "status": "incomplete",
    }
    assert context["research_coverage"]["by_destination"]["Osaka"]["activities"] == {
        "expected": 0,
        "usable": 0,
        "status": "not_requested",
    }
    assert context["research_by_destination"]["Osaka"]["activities"] == []


def test_prepare_context_rejects_missing_trip_length():
    with pytest.raises(ValueError, match="Trip length"):
        prepare_itinerary_context({
            "trip_intent": {"destinations": ["Tokyo"], "trip_days": None},
            "validated_jobs": [],
            "research_results": [],
        })


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def ainvoke(self, _messages):
        self.calls += 1
        return self.response


class FakePlanner:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **_kwargs):
        assert schema["title"] == "TripItinerary"
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        planning = json.loads(messages[1].content.split("research:\n", 1)[1])
        destinations = [
            destination for destination in planning["destinations"]
            for _ in range(planning["day_allocation"][destination])
        ]
        return TripItinerary(title="A plan", days=[
            {"destination": destination, "day_number": number}
            for number, destination in enumerate(destinations, start=1)
        ])


class FakeResearchGraph:
    def __init__(self, failure_destination=None):
        self.failure_destination = failure_destination

    async def ainvoke(self, state):
        request = state["messages"][0]["content"]
        destination = state["messages"][1].tool_calls[0]["args"]["destination"]
        if self.failure_destination in {"*", destination}:
            raise ValueError("unavailable")
        return {"messages": [SimpleNamespace(content=f"Research: {request}", type="ai")]}


def _workflow_context(failure_destination=None):
    return TripNodeContext(
        model=FakePlanner(),
        intent_model=FakeModel(
            TripIntent(
                destinations=["Tokyo", "Kyoto", "Osaka"],
                trip_days=7,
                needs_activities=True,
                needs_accommodation=False,
            )
        ),
        schemas={
            "search_attractions": {
                "type": "object",
                "properties": {"destination": {"type": "string"}},
                "required": ["destination"],
            },
            "search_hotels": {
                "type": "object",
                "properties": {"destination": {"type": "string"}},
                "required": ["destination"],
            },
        },
        activities_graph=FakeResearchGraph(failure_destination),
        accommodation_graph=FakeResearchGraph(),
    )


def test_workflow_prepares_context_once_after_research_fan_in(monkeypatch):
    calls = []
    original = nodes_module.prepare_itinerary_context

    def track_context(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(nodes_module, "prepare_itinerary_context", track_context)
    context = _workflow_context()
    graph = build_workflow(context)
    result = asyncio.run(graph.ainvoke(
        {
            "user_request": "Visit Tokyo, Kyoto, and Osaka over seven days",
            "issues": [],
            "input_errors": [],
            "job_answers": {},
            "plan_attempts": 0,
            "clarification_rounds": 0,
            "status": "pending",
        },
        config={"configurable": {"thread_id": "context-success"}},
    ))

    assert len(calls) == 1
    assert len(calls[0]["research_results"]) == 3
    assert result["itinerary_context"]["day_allocation"] == {
        "Tokyo": 3,
        "Kyoto": 2,
        "Osaka": 2,
    }
    assert context.model.calls == 1


def test_workflow_routes_partial_research_through_context(monkeypatch):
    calls = []
    original = nodes_module.prepare_itinerary_context

    def track_context(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(nodes_module, "prepare_itinerary_context", track_context)
    graph = build_workflow(_workflow_context(failure_destination="Tokyo"))
    result = asyncio.run(graph.ainvoke(
        {
            "user_request": "Visit Tokyo, Kyoto, and Osaka over seven days",
            "issues": [],
            "input_errors": [],
            "job_answers": {},
            "plan_attempts": 0,
            "clarification_rounds": 0,
            "status": "pending",
        },
        config={"configurable": {"thread_id": "context-partial"}},
    ))

    assert len(calls) == 1
    assert result["status"] == "partial"
    assert result["itinerary_context"]["research_coverage"]["failed"] == [
        "activities-0"
    ]


def test_workflow_keeps_terminal_research_failure_route(monkeypatch):
    calls = []
    original = nodes_module.prepare_itinerary_context

    def track_context(state):
        calls.append(state)
        return original(state)

    monkeypatch.setattr(nodes_module, "prepare_itinerary_context", track_context)
    graph = build_workflow(_workflow_context(failure_destination="*"))
    result = asyncio.run(graph.ainvoke(
        {
            "user_request": "Visit Tokyo, Kyoto, and Osaka over seven days",
            "issues": [],
            "input_errors": [],
            "job_answers": {},
            "plan_attempts": 0,
            "clarification_rounds": 0,
            "status": "pending",
        },
        config={"configurable": {"thread_id": "context-failure"}},
    ))

    assert result["status"] == "research_failed"
    assert calls == []
