import asyncio
from types import SimpleNamespace

import httpx
import pytest
from langgraph.errors import GraphInterrupt

from wander.graph.nodes.check_research_completeness import (
    create_check_research_completeness_node,
)
from wander.graph.nodes.research_worker import create_research_worker_node
from wander.graph.nodes.routes import route_research_completeness


class FakeGraph:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def ainvoke(self, _state):
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def worker_context(graph):
    return SimpleNamespace(
        activities_graph=graph,
        accommodation_graph=graph,
    )


def worker_state(agent="activities"):
    return {
        "job_id": "activities-0",
        "agent": agent,
        "destination": "Tokyo",
        "task": "Research attractions in Tokyo",
        "arguments": {"destination": "Tokyo"},
    }


def message_result(text="Useful research"):
    return {"messages": [SimpleNamespace(content=text, type="ai")]}


def test_success_first_attempt():
    graph = FakeGraph([message_result()])
    result = asyncio.run(create_research_worker_node(worker_context(graph))(worker_state()))
    record = result["research_results"][0]
    assert record["status"] == "completed"
    assert record["attempts"] == 1
    assert record["job_id"] == "activities-0"
    assert record["destination"] == "Tokyo"


def test_transient_retry(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    graph = FakeGraph([
        httpx.TimeoutException("timeout"),
        message_result("Recovered"),
    ])
    result = asyncio.run(create_research_worker_node(worker_context(graph))(worker_state()))
    record = result["research_results"][0]
    assert record["status"] == "completed"
    assert record["attempts"] == 2


def test_five_retryable_failures(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    graph = FakeGraph([httpx.ConnectError("down")])
    result = asyncio.run(create_research_worker_node(worker_context(graph))(worker_state()))
    record = result["research_results"][0]
    assert record["status"] == "failed"
    assert record["attempts"] == 5
    assert graph.calls == 5


def test_non_retryable_failure_no_retry():
    graph = FakeGraph([ValueError("invalid job arguments")])
    result = asyncio.run(create_research_worker_node(worker_context(graph))(worker_state()))
    record = result["research_results"][0]
    assert record["status"] == "failed"
    assert record["attempts"] == 1
    assert graph.calls == 1


def test_empty_and_explicit_no_data():
    empty = asyncio.run(create_research_worker_node(
        worker_context(FakeGraph([{"messages": []}]))
    )(worker_state()))
    assert empty["research_results"][0]["status"] == "no_data"

    explicit = asyncio.run(create_research_worker_node(
        worker_context(FakeGraph([{"status": "no_data"}]))
    )(worker_state()))
    assert explicit["research_results"][0]["status"] == "no_data"


def test_interrupt_and_cancellation_propagate():
    graph = FakeGraph([GraphInterrupt()])
    with pytest.raises(GraphInterrupt):
        asyncio.run(create_research_worker_node(worker_context(graph))(worker_state()))

    graph = FakeGraph([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(create_research_worker_node(worker_context(graph))(worker_state()))


def coverage(validated, results):
    node = create_check_research_completeness_node()
    return node({
        "validated_jobs": validated,
        "research_results": results,
    })["research_coverage"]


def test_completeness_all_failed_duplicate_missing_unknown():
    jobs = [
        {"job_id": "activities-0"},
        {"job_id": "accommodation-1"},
    ]
    results = [
        {"job_id": "activities-0", "status": "completed", "result": "Evidence"},
        {"job_id": "activities-0", "status": "completed", "result": "Duplicate"},
        {"job_id": "accommodation-1", "status": "failed", "result": None},
        {"job_id": "unknown-9", "status": "completed", "result": "Unknown"},
    ]
    coverage_value = coverage(jobs, results)
    assert coverage_value["completed"] == 1
    assert coverage_value["failed"] == ["accommodation-1"]
    assert coverage_value["duplicate"] == ["activities-0"]
    assert coverage_value["unknown"] == ["unknown-9"]


def test_completeness_no_data_and_missing():
    coverage_value = coverage(
        [{"job_id": "activities-0"}, {"job_id": "accommodation-1"}],
        [{"job_id": "activities-0", "status": "no_data", "result": None}],
    )
    assert coverage_value["no_data"] == ["activities-0"]
    assert coverage_value["missing"] == ["accommodation-1"]


def test_routing_with_partial_and_no_research():
    partial = {
        "research_coverage": {
            "completed": 5,
            "failed": ["accommodation-1"],
            "no_data": [],
            "missing": [],
            "duplicate": [],
            "unknown": [],
        }
    }
    assert route_research_completeness(partial) == "prepare_itinerary_context"

    none = {
        "research_coverage": {
            "completed": 0,
            "failed": ["activities-0"],
            "no_data": [],
            "missing": [],
            "duplicate": [],
            "unknown": [],
        }
    }
    assert route_research_completeness(none) == "needs_input"
