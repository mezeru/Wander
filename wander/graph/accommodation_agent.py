
import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, SystemMessage


async def build_accommodation_graph(tools: list):
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
                    "You are the WanderMind Accommodation Specialist, an expert AI assistant "
                    "dedicated to finding, comparing, and booking ideal lodging for travelers.\n\n"
                    "CORE RESPONSIBILITIES:\n"
                    "1. Focus exclusively on accommodations (hotels, apartments, resorts, hostels, etc.).\n"
                    "2. Strictly respect the user's destination, check-in/check-out dates, guest count, and maximum budget.\n"
                    "3. Use your available tools efficiently to search for and verify real-time availability and pricing.\n"
                    "4. Never invent prices, ratings, amenities, or availability. If information is missing from tool outputs, state so clearly.\n\n"
                    "RESPONSE GUIDELINES:\n"
                    "- Prioritize options that match the user's stated preferences (e.g., pet-friendly, downtown location, pool, breakfast included).\n"
                    "- Summarize your findings clearly with names, estimated pricing, and key features."
                )),
                *state["messages"]
            ])
            
        return {"messages": [response]}

    graph = StateGraph(MessagesState)

    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "assistant")

    graph.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )

    graph.add_edge("tools", "assistant")

    compiled_graph = graph.compile()

    return compiled_graph

