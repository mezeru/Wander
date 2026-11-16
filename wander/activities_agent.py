import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter
from langchain.agents import create_agent


async def main():

    server_path = Path(__file__).with_name("mcp_server.py")

    async with MCPAdapter(server_path) as adapter:

        tools = await adapter.list_tools()

        # Inspect discovered tools
        for tool in tools:
            print("Tool:", tool.name)
            print("Arguments:", tool.args)

        agent = create_agent(
            model="openai:gpt-5.5",
            tools=tools,
            system_prompt=(
                "You are an Activities Agent for WanderMind AI. "
                "Use available travel tools to research attractions. "
                "Do not invent attraction information or claim that "
                "mock travel data is live."
            )
        )

        response = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find historical and cultural attractions "
                        "in Kyoto for a two-day trip."
                    )
                }
            ]
        })

        for message in response["messages"]:
            print(f"\n{type(message).__name__}")
            print(message.content)

            if getattr(message, "tool_calls", None):
                print("Tool calls:", message.tool_calls)


if __name__ == "__main__":
    asyncio.run(main())