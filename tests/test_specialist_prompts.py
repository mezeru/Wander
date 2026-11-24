import asyncio
import json
import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from wander.graph import activities_graph, accommodation_agent
from wander.graph.nodes.context import TripNodeContext
from wander.graph.nodes.final_planner import create_final_planner_node
from wander.graph.nodes.routes import route_review
from wander.graph.nodes.validate_final import create_validate_final_node
from wander.graph.schemas import ItineraryDay, PlannedActivity, TripItinerary


class FakeToolCallingModel:
    def __init__(self, tool_name, final_answer):
        self.tool_name = tool_name
        self.final_answer = final_answer
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": self.tool_name,
                    "args": {"destination": "Kyoto"},
                    "id": "test-call",
                }],
            )
        self.tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        return AIMessage(content=self.final_answer)


def test_activities_agent_tool_loop_preserves_multiple_real_attractions(monkeypatch):
    @tool
    def search_attractions(destination: str) -> str:
        """Return sightseeing attractions for a destination."""
        return json.dumps({
            "attractions": [
                {"name": "Kiyomizu-dera"},
                {"name": "Fushimi Inari Taisha"},
            ]
        })

    model = FakeToolCallingModel(
        "search_attractions",
        "Shortlist: Kiyomizu-dera and Fushimi Inari Taisha.",
    )
    monkeypatch.setattr(activities_graph, "ChatOllama", lambda **_kwargs: model)

    graph = asyncio.run(activities_graph.build_activities_graph([search_attractions]))
    result = asyncio.run(graph.ainvoke({
        "messages": [HumanMessage(content="Research sightseeing in Kyoto.")]
    }))

    prompt = model.calls[0][0].content
    tool_output = model.tool_messages[0].content
    final_answer = result["messages"][-1].content
    assert model.bound_tools == [search_attractions]
    assert {"assistant", "tools"} <= set(graph.get_graph().nodes)
    assert "multiple relevant attractions" in prompt
    assert "preserve the actual attraction names" in prompt
    assert "Do not invent attractions" in prompt
    assert "Kiyomizu-dera" in tool_output
    assert "Fushimi Inari Taisha" in tool_output
    assert "Kiyomizu-dera" in final_answer
    assert "Fushimi Inari Taisha" in final_answer


def test_accommodation_agent_marks_unknown_prices_and_availability(monkeypatch):
    @tool
    def search_hotels(destination: str) -> str:
        """Return hotel research for a destination."""
        return json.dumps({
            "hotels": [
                {
                    "name": "Hilton Kyoto",
                    "nightly_price": None,
                    "availability_verified": False,
                },
                {
                    "name": "Hotel Gran Ms Kyoto",
                    "nightly_price": None,
                    "availability_verified": False,
                },
            ]
        })

    model = FakeToolCallingModel(
        "search_hotels",
        "Hilton Kyoto and Hotel Gran Ms Kyoto. Prices and availability are unknown.",
    )
    monkeypatch.setattr(accommodation_agent, "ChatOllama", lambda **_kwargs: model)

    graph = asyncio.run(accommodation_agent.build_accommodation_graph([search_hotels]))
    result = asyncio.run(graph.ainvoke({
        "messages": [HumanMessage(content="Find hotels in Kyoto under $150.")]
    }))

    prompt = model.calls[0][0].content
    tool_output = model.tool_messages[0].content
    final_answer = result["messages"][-1].content
    assert model.bound_tools == [search_hotels]
    assert {"assistant", "tools"} <= set(graph.get_graph().nodes)
    assert "each distinct hotel" in prompt
    assert "without price data" in prompt
    assert "Never claim that date-specific availability was checked" in prompt
    assert "nightly_price" in tool_output
    assert "availability_verified" in tool_output
    assert final_answer.count("Hotel Gran Ms Kyoto") == 1
    assert "unknown" in final_answer
    assert "within budget" not in final_answer


class FakePlannerResponse:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return TripItinerary(
            title="Sparse Kyoto itinerary",
            days=[
                ItineraryDay(
                    day_number=1,
                    destination="Kyoto",
                    activities=[
                        PlannedActivity(
                            name="Nashinoki Shrine",
                            evidence_job_id="activities-0",
                        )
                    ],
                ),
                ItineraryDay(day_number=2, destination="Kyoto"),
                ItineraryDay(day_number=3, destination="Kyoto"),
            ],
            warnings=["Only one researched attraction was available for three days."],
        )


class FakeStructuredPlanner:
    def __init__(self):
        self.response = FakePlannerResponse()

    def with_structured_output(self, _schema, method):
        assert method == "json_schema"
        return self.response


def test_sparse_research_is_warned_without_a_revision_loop(caplog):
    model = FakeStructuredPlanner()
    context = TripNodeContext(model=model)
    state = {
        "user_request": "Plan a 3-day trip to Kyoto.",
        "trip_intent": {
            "destinations": ["Kyoto"],
            "trip_days": 3,
            "needs_activities": True,
            "needs_accommodation": False,
        },
        "research_results": [{
            "job_id": "activities-0",
            "agent": "activities",
            "destination": "Kyoto",
            "status": "completed",
            "result": {"attractions": [{"name": "Nashinoki Shrine"}]},
        }],
        "validated_jobs": [{
            "job_id": "activities-0",
            "agent": "activities",
            "destination": "Kyoto",
        }],
        "research_coverage": {"expected": 1, "completed": 1},
        "itinerary_context": {
            "destinations": ["Kyoto"],
            "total_days": 3,
            "day_allocation": {"Kyoto": 3},
        },
        "plan_attempts": 0,
    }

    planned = asyncio.run(create_final_planner_node(context)(state))
    state.update(planned)
    with caplog.at_level(logging.INFO, logger="wander.graph.nodes.validate_final"):
        validation = asyncio.run(
            create_validate_final_node(SimpleNamespace())(state)
        )
    state.update(validation)

    planner_messages = model.response.messages
    prompt = planner_messages[0].content
    assert "distribute them across days where possible" in prompt
    assert "Never invent attractions to fill the schedule" in prompt
    assert "warnings" in prompt
    assert len(planned["structured_plan"]["days"]) == 3
    assert planned["structured_plan"]["days"][1]["activities"] == []
    assert validation["issues"] == []
    assert validation["status"] == "partial"
    assert any("Only one researched attraction" in item for item in validation["research_warnings"])
    assert "attempt=1 status=partial correctable_issues=[]" in caplog.text
    assert route_review({
        **state,
        "issues": validation["issues"],
        "status": validation["status"],
        "plan_attempts": 1,
    }) == "__end__"
