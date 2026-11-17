from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from .activities_graph import build_activities_graph
from .accommodation_agent import build_accommodation_graph
from pathlib import Path
import asyncio


from typing import TypedDict
from pydantic import BaseModel, Field


class ResearchTasks(BaseModel):
    activities_task: str = Field(
        description="Instructions for researching trip activities"
    )
    accommodation_task: str = Field(
        description="Instructions for researching accommodation"
    )


class TripState(TypedDict):
    user_request: str

    activities_task: str
    accommodation_task: str

    activities_result: str
    accommodation_result: str

    final_plan: str

model = ChatOllama(
            model="qwen3:4b",
            temperature=0
        )

async def main():
    server_path = (
        Path(__file__).resolve().parent.parent
        / "mcp_server.py"
    )

    async with MCPAdapter(server_path) as adapter:
        all_tools = await adapter.list_tools()

        activities_tools = [
            tool for tool in all_tools
            if tool.name == "search_attractions"
        ]

        hotel_tools = [
            tool for tool in all_tools
            if tool.name == "search_hotels"
        ]

        activities_graph = await build_activities_graph(activities_tools)
        accommodation_graph = await build_accommodation_graph(hotel_tools)

        structured_model = model.with_structured_output(ResearchTasks)

        async def coordinator_node(state: TripState):
            response = await structured_model.ainvoke([
                SystemMessage(content=(
                    "You are the WanderMind Trip Coordinator. "
                    "Break the user's request into two independent "
                    "research tasks: activities and accommodation. "
                    "Preserve destinations, dates, preferences and "
                    "budget constraints. Do not invent missing details."
                )),
                HumanMessage(content=state["user_request"])
            ])

            return response.model_dump()


        async def activities(state: TripState):

            result = await activities_graph.ainvoke({
                "messages": [
                    {
                        "role": "user",
                        "content": state["activities_task"]
                    }
                ]
            })

            return {
                "activities_result": result["messages"][-1].content
            }


        async def accomodation(state: TripState):
            result = await accommodation_graph.ainvoke({
                "messages": [
                    {
                        "role": "user",
                        "content": state["accommodation_task"]
                    }
                ]
            })
            return {
                "accommodation_result": result["messages"][-1].content
            }

        async def final_planner(state: TripState):
            response = await model.ainvoke([
                SystemMessage(content=(
                    "You are a travel itinerary planner. "
                    "Combine the research into a practical trip. "
                    "Respect the user's dates and budget. "
                    "Do not invent availability or prices. "
                    "Clearly label all mock estimates."
                )),
                HumanMessage(content=f"""
        Original request:
        {state["user_request"]}

        Activities research:
        {state["activities_result"]}

        Accommodation research:
        {state["accommodation_result"]}
        """)
            ])

            return {"final_plan": response.content}



        graph = StateGraph(TripState)

        graph.add_node("coordinator",coordinator_node)
        graph.add_node("activities",activities)
        graph.add_node("accommodation",accomodation)
        graph.add_node("final_planner",final_planner)

        graph.add_edge(START,"coordinator")
        graph.add_edge("coordinator", "activities")
        graph.add_edge("activities", "accommodation")
        graph.add_edge("accommodation", "final_planner")
        graph.add_edge("final_planner", END)

        compiled_graph = graph.compile()

        result = await compiled_graph.ainvoke({
            "user_request": (
                "Plan a 3-day Kyoto trip with a nightly "
                "hotel budget of $150."
            )
        })

        print(result["final_plan"])

if __name__ == "__main__":
    asyncio.run(main())



