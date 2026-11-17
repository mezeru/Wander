
import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, SystemMessage


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
            response = await model_with_tools.ainvoke(
                    SystemMessage(content=(
                "Your tool budget is exhausted. "
                "Provide a final answer using only "
                "information already retrieved. "
                "Do not invent missing information."
            )),
            *state["messages"])

        else:
            response = await model_with_tools.ainvoke(
                state["messages"]
            )
            
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
