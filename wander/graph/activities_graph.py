
import asyncio
import logging
from pathlib import Path

from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, SystemMessage

from .logging import log_node


logger = logging.getLogger(__name__)


async def build_activities_graph(tools: list):


    model = ChatOllama(
        model="qwen3:4b",
        temperature=0
    )

    MAX_TOOL_CALLS = 3
    model_with_tools = model.bind_tools(tools)

    def count_tool_calls(state: MessagesState) -> int:
        return sum(
            isinstance(message, ToolMessage)
            for message in state["messages"]
        )

    async def assistant(state: MessagesState):

        calls_used = count_tool_calls(state)

        if calls_used >= MAX_TOOL_CALLS:
            response = await model.ainvoke(
                    SystemMessage(content=(
                '''Your tool budget is exhausted. 
                Provide a final answer using only 
                information already retrieved. 
                Do not invent missing information.'''
            )),
            *state["messages"])

        else:
            response = await model_with_tools.ainvoke([
                SystemMessage(content=(
                    "You are the WanderMind Activities Specialist, an expert AI assistant "
                    "dedicated to discovering, scheduling, and recommending top-tier activities, "
                    "tours, attractions, and local experiences for travelers.\n\n"
                    "CORE RESPONSIBILITIES:\n"
                    "1. Focus exclusively on activities, entertainment, sightseeing, dining spots, and local experiences.\n"
                    "2. Strictly respect the user's destination, available dates/times, party size, and budget limits.\n"
                    "3. Use your available tools efficiently to search for real-time ticket availability, operating hours, and pricing.\n"
                    "4. Never invent ticket prices, operating hours, or ratings. If information is missing from tool outputs, state so clearly.\n\n"
                    "RESPONSE GUIDELINES:\n"
                    "- Prioritize options matching the user's stated interests (e.g., family-friendly, outdoor adventure, historical, culinary).\n"
                    "- Present recommendations with clear names, estimated costs or time durations, and why they fit the traveler's goals."
                )),
                *state["messages"]
            ])
            
        return {"messages": [response]}

    def route_assistant(state: MessagesState) -> str:
        destination = tools_condition(state)
        logger.info("Workflow route: activities.assistant -> %s", destination)
        return destination

    graph = StateGraph(MessagesState)

    tool_node = ToolNode(tools)
    graph.add_node("assistant", log_node("activities.assistant", assistant))
    graph.add_node("tools", log_node("activities.tools", tool_node.ainvoke))

    graph.add_edge(START, "assistant")

    graph.add_conditional_edges(
        "assistant",
        route_assistant,
        {
            "tools": "tools",
            "__end__": END
        }
    )

    graph.add_edge("tools", "assistant")

    compiled_graph = graph.compile()

    return compiled_graph
