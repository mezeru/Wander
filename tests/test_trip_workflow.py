import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import tool
from langgraph.types import Command

import wander.graph.nodes as trip_nodes
from wander.graph.accommodation_agent import build_accommodation_graph
from wander.graph.activities_graph import build_activities_graph
from wander.graph.nodes.context import TripNodeContext
from wander.graph.nodes.routes import MAX_CLARIFICATION_ROUNDS, MAX_PLAN_ATTEMPTS
from wander.graph.nodes.trip_intent import create_extract_trip_intent_node
from wander.graph.schemas import TripIntent, TripItinerary
from wander.graph.specialist import SpecialistSummary
from wander.graph.trip_coordinator import DEMO_REQUEST, build_workflow, read_trip_request, tool_schema


class IntentModel:
    def __init__(self, **expected):
        self.response = TripIntent(**expected)
        self.requests = []

    async def ainvoke(self, messages):
        self.requests.append(messages[-1].content)
        return self.response


class Models:
    def __init__(self, defective_attempts=0, malformed_summary=False):
        self.defective_attempts = defective_attempts
        self.malformed_summary = malformed_summary
        self.planner_calls = []
        self.summaries = []

    def with_structured_output(self, schema, **_kwargs):
        async def invoke(messages):
            if schema is SpecialistSummary:
                self.summaries.append(schema)
                if self.malformed_summary:
                    raise OutputParserException("Malformed model summary")
                # The real model may omit notes for most candidates. They must survive.
                return schema(places=[{"number": 1, "nightly_rate": 100, "availability_verified": True}])

            assert schema["title"] == "TripItinerary"
            self.planner_calls.append(messages)
            raw = messages[1].content.split("research:\n", 1)[1]
            context = json.loads(raw)
            allocation = context["day_allocation"]
            counts = context["research_by_destination"]
            days = []
            for destination in context["destinations"]:
                attractions = [
                    {"name": entry["name"], "evidence_job_id": report["job_id"]}
                    for report in counts[destination]["activities"]
                    for entry in report["result"]["attractions"]
                ]
                lodging = [
                    (entry["name"], report["job_id"])
                    for report in counts[destination]["accommodation"]
                    for entry in report["result"]["hotels"]
                ]
                for offset in range(allocation[destination]):
                    day = {"day_number": len(days) + 1, "destination": destination}
                    if offset < len(attractions):
                        day["activities"] = [attractions[offset]]
                    if lodging and offset == 0:
                        day["accommodation"], day["accommodation_evidence_job_id"] = lodging[0]
                    days.append(day)
            if len(self.planner_calls) <= self.defective_attempts:
                days[0]["activities"] = [
                    activity for day in days for activity in day.get("activities", [])
                ]
                for day in days[1:]:
                    day["activities"] = []
            return TripItinerary(title="Test itinerary", days=days)

        return SimpleNamespace(ainvoke=invoke)


def model_intent(destinations, days=None, activities=True, lodging=True, **kwargs):
    return IntentModel(
        destinations=destinations, trip_days=days, needs_activities=activities,
        needs_accommodation=lodging, **kwargs,
    )


async def make_workflow(
    intent, *, activity_count=3, worker_batch=0, live=False, models=None,
):
    calls = []
    started = asyncio.Event()
    models = models or Models()

    async def record(name, destination, **arguments):
        calls.append((name, destination, arguments))
        if worker_batch:
            if len(calls) >= worker_batch:
                started.set()
            await asyncio.wait_for(started.wait(), timeout=3)

    @tool(response_format="content_and_artifact")
    async def search_attractions(destination: str):
        """Discover attraction candidates."""
        await record("attractions", destination)
        places = [
            {"id": f"{destination}-attraction-{i}", "name": f"{destination} Attraction {i}"}
            for i in range(activity_count)
        ]
        return json.dumps(places), {"structured_content": {"result": places}}

    @tool(response_format="content_and_artifact")
    async def search_hotels(destination: str, check_in: str, check_out: str, budget: float):
        """Discover accommodation candidates."""
        await record("hotels", destination, check_in=check_in, check_out=check_out, budget=budget)
        places = [
            {"id": f"{destination}-A", "name": f"{destination} Hotel",
             "nightly_price": None, "availability_verified": False},
            {"id": f"{destination}-alias", "name": f"{destination.upper()} HOTEL"},
            {"id": f"{destination}-A", "name": f"{destination} Alternate Name"},
            {"id": f"{destination}-B", "name": f"{destination} Guesthouse"},
        ]
        return json.dumps(places), {"structured_content": {"result": places}}

    context = TripNodeContext(
        model=models,
        intent_model=intent,
        live_providers=live,
        schemas={t.name: tool_schema(t) for t in (search_attractions, search_hotels)},
        activities_graph=await build_activities_graph([search_attractions], models),
        accommodation_graph=await build_accommodation_graph([search_hotels], models),
    )
    return build_workflow(context), calls, models


def config():
    return {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 50}


def fields(result):
    return {
        (field["job_id"], field["field"])
        for field in result["__interrupt__"][0].value["fields"]
    }


@pytest.mark.parametrize(
    ("request_text", "intent", "expected"),
    [
        ("Plan a 3-day trip to Kyoto.", model_intent(["Kyoto"], activities=False),
         (["Kyoto"], 3, True, True, None, True)),
        ("Plan a five-day trip to Kyoto and Tokyo.", model_intent(["Kyoto", "Tokyo"]),
         (["Kyoto", "Tokyo"], 5, True, True, None, True)),
        ("Find hotels in Tokyo.", model_intent(["Tokyo"], 3),
         (["Tokyo"], None, False, True, None, False)),
        ("Find hotels in Tokyo for my itinerary.", model_intent(["Tokyo"], 3),
         (["Tokyo"], None, False, True, None, False)),
        ("Show me attractions in Paris.", model_intent(["Paris"]),
         (["Paris"], None, True, False, None, False)),
        ("Plan a trip to Osaka with a total budget of $500.",
         model_intent(["Osaka"], nightly_hotel_budget=500),
         (["Osaka"], None, True, True, None, True)),
        ("Find hotels in Tokyo for GBP 150 per night.", model_intent(["Tokyo"]),
         (["Tokyo"], None, False, True, None, False)),
        ("Plan a two-day trip to Chickasha, Oklahoma.", model_intent(["Chickasha, Oklahoma"]),
         (["Chickasha, Oklahoma"], 2, True, True, None, True)),
        ("Plan a trip to Kyoto for 5 days. Hotel budget is $150 per night.",
         model_intent(["Kyoto"]), (["Kyoto"], 5, True, True, 150, True)),
        ("Spend two days in Kyoto.", model_intent(["Kyoto"], lodging=False),
         (["Kyoto"], 2, True, False, None, True)),
        ("Plan a trip to Kyoto for 3 or 5 days.", model_intent(["Kyoto"], 3),
         (["Kyoto"], None, True, True, None, True)),
        ("Plan a trip to Kyoto. I need a transit pass for 5 days.", model_intent(["Kyoto"]),
         (["Kyoto"], None, True, True, None, True)),
    ],
)
def test_intent_uses_original_text_and_recovers_only_clear_values(request_text, intent, expected):
    text = f"  {request_text}  "
    context = TripNodeContext(intent_model=intent)
    result = asyncio.run(create_extract_trip_intent_node(context)({"user_request": text}))["trip_intent"]
    assert intent.requests == [text]
    assert tuple(result[key] for key in (
        "destinations", "trip_days", "needs_activities", "needs_accommodation",
        "nightly_hotel_budget", "requires_itinerary",
    )) == expected


def test_cli_preserves_input_and_announces_blank_demo(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "  Find hotels in Tokyo.  ")
    assert read_trip_request() == "  Find hotels in Tokyo.  "
    assert capsys.readouterr().out == ""
    monkeypatch.setattr("builtins.input", lambda _: " \t ")
    assert read_trip_request() == DEMO_REQUEST
    assert "Using the Kyoto demo" in capsys.readouterr().out


def test_complete_kyoto_workflow_asks_only_missing_dates_and_plans_three_days():
    async def exercise():
        request = (
            "Plan a 3-day trip to Kyoto, Japan. Include sightseeing attractions and "
            "accommodation. My nightly hotel budget is $150."
        )
        intent = model_intent(["Kyoto"], activities=False)
        graph, calls, models = await make_workflow(intent, worker_batch=2)
        thread = config()
        first = await graph.ainvoke({"user_request": request}, thread)
        assert fields(first) == {
            ("accommodation-0", "check_in"), ("accommodation-0", "check_out"),
        }
        assert first["trip_intent"]["trip_days"] == 3
        assert not calls
        result = await graph.ainvoke(Command(resume={"accommodation-0": {
            "check_in": "2027-04-10", "check_out": "04/12/2027",
        }}), thread)
        assert sorted(call[0] for call in calls) == ["attractions", "hotels"]
        assert result["research_coverage"]["finished"] == 2
        assert result["plan_attempts"] == 1 == len(models.planner_calls)
        assert result["status"] == "partial"  # Live room availability was never established.
        assert result["issues"] == []
        assert [slot["lodging_night"] for slot in result["itinerary_context"]["day_slots"]] == [
            True, True, False,
        ]
        days = result["structured_plan"]["days"]
        assert [day["day_number"] for day in days] == [1, 2, 3]
        assert len({day["activities"][0]["name"] for day in days}) == 3
        assert days[0]["accommodation"] == "Kyoto Hotel"
        assert days[0]["accommodation_evidence_job_id"] == "accommodation-0"
        assert any("budget fit" in warning for warning in result["research_warnings"])
        assert len(intent.requests) == 1

    asyncio.run(exercise())


def test_global_and_job_clarifications_preserve_state_and_dispatch_four_jobs_once():
    async def exercise():
        intent = model_intent(["Kyoto", "Tokyo"])
        graph, calls, models = await make_workflow(intent, worker_batch=4)
        thread = config()
        request = "Plan a sightseeing trip to Kyoto and Tokyo. Include hotels."
        result = await graph.ainvoke({"user_request": request}, thread)
        assert fields(result) == {("__trip__", "trip_days")}
        assert "research_jobs" not in result
        assert not calls
        result = await graph.ainvoke(Command(resume={"__trip__": {"trip_days": True}}), thread)
        assert fields(result) == {("__trip__", "trip_days")}
        assert not calls
        result = await graph.ainvoke(Command(resume={"__trip__": {"trip_days": "5"}}), thread)
        original_jobs = result["research_jobs"]
        assert len(original_jobs) == len(fields(result)) - 2 == 4
        assert not calls
        result = await graph.ainvoke(Command(resume={
            "accommodation-0": {"check_in": "2027-04-10", "check_out": "2027-04-13", "budget": "$150"},
            "accommodation-1": {"check_in": "2027-04-13", "check_out": "tomorrow", "budget": 180},
        }), thread)
        assert fields(result) == {("accommodation-1", "check_out")}
        assert not calls
        result = await graph.ainvoke(Command(resume={
            "accommodation-1": {"check_out": "2027-04-15", "budget": 1},
        }), thread)
        assert result["research_jobs"] == original_jobs
        assert result["trip_intent"]["destinations"] == ["Kyoto", "Tokyo"]
        assert result["trip_intent"]["trip_days"] == 5
        assert result["job_answers"]["accommodation-1"]["budget"] == 180
        assert len({record["job_id"] for record in result["research_results"]}) == 4
        assert len(calls) == 4 and len(intent.requests) == 1
        assert len(models.planner_calls) == 1
        assert [day["destination"] for day in result["structured_plan"]["days"]] == [
            "Kyoto", "Kyoto", "Kyoto", "Tokyo", "Tokyo",
        ]
        assert result["structured_plan"]["days"][3]["transfer_notes"] == (
            "Transfer from Kyoto to Tokyo. Check transport options and journey time."
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("count", [0, 1])
def test_sparse_attractions_are_partial_without_futile_revisions(count):
    async def exercise():
        intent = model_intent(["Kyoto"], lodging=False)
        graph, calls, models = await make_workflow(intent, activity_count=count)
        result = await graph.ainvoke({"user_request": "Plan a 3-day sightseeing trip to Kyoto."}, config())
        assert result["status"] == "partial"
        assert result["issues"] == []
        assert len(calls) == result["plan_attempts"] == 1
        assert len(models.summaries) == count
        assert sum(len(day["activities"]) for day in result["structured_plan"]["days"]) == count
        assert any(f"{count} distinct researched attractions" in warning
                   for warning in result["research_warnings"])

    asyncio.run(exercise())


def test_lodging_lookup_needs_no_duration_preserves_ids_and_keeps_rate_unknown():
    async def exercise():
        intent = model_intent(["Tokyo"], activities=False)
        models = Models()
        graph, calls, _ = await make_workflow(intent, live=True, models=models)
        thread = config()
        request = "Find hotels in Tokyo. My hotel budget is $150 per night."
        result = await graph.ainvoke({"user_request": request}, thread)
        assert fields(result) == {("accommodation-0", "check_in"), ("accommodation-0", "check_out")}
        result = await graph.ainvoke(Command(resume={"accommodation-0": {
            "check_in": "2027-04-10", "check_out": "2027-04-12",
        }}), thread)
        assert result["status"] == "partial"
        assert len(calls) == 1 and not models.planner_calls
        assert not result.get("structured_plan")
        assert result["trip_intent"]["trip_days"] is None
        report = result["research_results"][0]["result"]["hotels"]
        assert [(hotel["id"], hotel["name"]) for hotel in report] == [
            ("Tokyo-A", "Tokyo Hotel"), ("Tokyo-B", "Tokyo Guesthouse"),
        ]
        assert all(hotel["nightly_price"] is None and hotel["nightly_rate"] is None
                   and hotel["availability_verified"] is False for hotel in report)
        assert any("budget fit" in warning for warning in result["research_warnings"])

    asyncio.run(exercise())


@pytest.mark.parametrize("defective", [1, MAX_PLAN_ATTEMPTS])
def test_planner_revises_omitted_coverage_with_explicit_feedback_and_cap(defective):
    async def exercise():
        models = Models(defective_attempts=defective)
        graph, calls, _ = await make_workflow(model_intent(["Kyoto"], lodging=False), models=models)
        result = await graph.ainvoke({"user_request": "Plan a 3-day sightseeing trip to Kyoto."}, config())
        attempts = min(defective + 1, MAX_PLAN_ATTEMPTS)
        assert result["plan_attempts"] == len(models.planner_calls) == attempts
        assert len(calls) == 1
        assert "across 3 different days" in models.planner_calls[1][-1].content
        assert result["status"] == ("verified" if defective < MAX_PLAN_ATTEMPTS else "invalid")

    asyncio.run(exercise())


@pytest.mark.parametrize("initial_days", [None, -1, 2.5, True])
def test_invalid_global_duration_and_missing_destination_never_dispatch(initial_days):
    async def exercise():
        intent = model_intent([], initial_days)
        graph, calls, _ = await make_workflow(intent)
        thread = config()
        result = await graph.ainvoke({"user_request": "Plan a trip."}, thread)
        assert fields(result) == {("__trip__", "trip_days"), ("__trip__", "destinations")}
        for _ in range(MAX_CLARIFICATION_ROUNDS):
            result = await graph.ainvoke(Command(resume={"__trip__": {"trip_days": ""}}), thread)
        assert result["status"] == "needs_input"
        assert len(result["missing_details"]) == 2
        assert not calls
        assert len(intent.requests) == 1

    asyncio.run(exercise())


def test_unsupported_request_terminates_with_explanation_and_no_empty_interrupt():
    async def exercise():
        intent = model_intent([], activities=False, lodging=False, requires_itinerary=False)
        graph, calls, _ = await make_workflow(intent)
        result = await graph.ainvoke({"user_request": "What is two plus two?"}, config())
        assert result["status"] == "no_research"
        assert result["status_message"]
        assert not calls and not result.get("__interrupt__")

    asyncio.run(exercise())


def test_empty_jobs_after_complete_intent_is_a_workflow_error(monkeypatch):
    async def empty_jobs(_state):
        return {"research_jobs": []}

    monkeypatch.setattr(trip_nodes, "create_research_jobs_node", lambda _context: empty_jobs)

    async def exercise():
        graph, calls, _ = await make_workflow(model_intent(["Kyoto"], lodging=False))
        result = await graph.ainvoke({"user_request": "Plan a 3-day sightseeing trip to Kyoto."}, config())
        assert result["status"] == "workflow_error"
        assert "No research jobs" in result["status_message"]
        assert not calls and not result.get("__interrupt__")

    asyncio.run(exercise())


def test_one_day_for_two_stops_requests_clarification_before_job_creation():
    async def exercise():
        graph, calls, _ = await make_workflow(model_intent(["Kyoto", "Tokyo"], lodging=False))
        result = await graph.ainvoke({"user_request": "Plan a 1-day trip to Kyoto and Tokyo."}, config())
        assert fields(result) == {("__trip__", "trip_days")}
        assert "at least 2" in result["missing_details"][0]["description"]
        assert not calls and "research_jobs" not in result

    asyncio.run(exercise())


def test_attraction_lookup_resumes_missing_destination_and_preserves_places_on_invalid_summary():
    async def exercise():
        intent = model_intent([], lodging=False)
        models = Models(malformed_summary=True)
        graph, calls, _ = await make_workflow(intent, models=models)
        thread = config()
        first = await graph.ainvoke({"user_request": "Show me attractions."}, thread)
        assert fields(first) == {("__trip__", "destinations")}
        assert not calls
        result = await graph.ainvoke(Command(resume={"__trip__": {"destinations": "Paris"}}), thread)
        assert result["status"] == "research_complete"
        assert not models.planner_calls
        report = result["research_results"][0]["result"]
        assert [place["id"] for place in report["attractions"]] == [
            "Paris-attraction-0", "Paris-attraction-1", "Paris-attraction-2",
        ]
        assert report["limitations"] == ["Relevance notes were unavailable."]
        assert len(calls) == len(intent.requests) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("correction", "expected_days"),
    [("5", {"Kyoto": 3, "Tokyo": 2}),
     ("accommodation-0.check_in=2027-04-12", {"Kyoto": 1, "Tokyo": 2})],
)
def test_explicit_hotel_dates_conflicting_span_can_be_corrected_without_recreating_jobs(correction, expected_days):
    async def exercise():
        stays = [
            {"destination": "Kyoto", "check_in": "2027-04-10", "check_out": "2027-04-13"},
            {"destination": "Tokyo", "check_in": "2027-04-13", "check_out": "2027-04-15"},
        ]
        request = (
            "Plan a 3-day trip to Kyoto and Tokyo. Include hotels with a $150 per night budget. "
            "Kyoto check-in 2027-04-10 and check-out 2027-04-13. "
            "Tokyo check-in 2027-04-13 and check-out 2027-04-15."
        )
        graph, calls, _ = await make_workflow(model_intent(["Kyoto", "Tokyo"], hotel_stays=stays))
        thread = config()
        first = await graph.ainvoke({"user_request": request}, thread)
        assert fields(first) == {("__trip__", "trip_days")}
        assert not calls
        result = await graph.ainvoke(Command(resume={"__trip__": {"trip_days": correction}}), thread)
        assert result["research_jobs"] == first["research_jobs"]
        assert sorted((call[1], call[2]["check_out"]) for call in calls if call[0] == "hotels") == [
            ("Kyoto", "2027-04-13"), ("Tokyo", "2027-04-15"),
        ]
        assert result["itinerary_context"]["day_allocation"] == expected_days

    asyncio.run(exercise())
