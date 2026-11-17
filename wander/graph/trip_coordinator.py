from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from .activities_graph import build_activities_graph
from .accommodation_agent import build_accommodation_graph
from pathlib import Path


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
    server_path = Path(__file__).with_name("mcp_server.py")

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

        activities_graph = build_activities_graph(activities_tools)
        accommodation_graph = build_accommodation_graph(hotel_tools)

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
                        "content": state["accomodation_task"]
                    }
                ]
            })
            return {
                "accomodation_result": result["messages"][-1].content
            }

        async def final_planner(state: TripState):
            # TODO 1: Get the original user request

            # TODO 2: Get activities_result

            # TODO 3: Get accommodation_result

            # TODO 4: Ask the LLM to combine the results
            # into a practical itinerary.

            # TODO 5: Do not invent missing travel information.
            # Label all mock prices and availability clearly.

            # TODO 6: Return:
            # {"final_plan": response.content}

            pass



        graph = StateGraph(TripState)

        graph.add_node("coordinator",coordinator_node)
        graph.add_node("activities",activities)
        graph.add_node("accomodation",accomodation)
        graph.add_node("final_planner",final_planner)

        graph.add_edge(START,"coordinator")
        # TODO Add other edges for delegation
        graph.add_edge("final_planner",END)



