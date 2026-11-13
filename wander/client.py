from __future__ import annotations

import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    server = StdioServerParameters(
        command="python",
        args=["-m", "wander.mcp_server"],
        cwd=str(project_root),
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_list = await session.list_tools()
            print("Available tools:")
            for tool in tool_list.tools:
                print(f"- {tool.name}: {tool.description}")

            prompt_list = await session.list_prompts()
            print("\nAvailable prompts:")
            for prompt in prompt_list.prompts:
                print(f"- {prompt.name}: {prompt.description}")

            result = await session.call_tool(
                "search_attractions",
                {
                    "destination": "Tokyo",
                    "interests": ["food", "culture"],
                    "limit": 2,
                },
            )
            print("\nAttraction results:")
            print(result.model_dump())

            hotel_result = await session.call_tool(
                "search_hotels",
                {
                    "destination": "Kyoto",
                    "check_in": "2026-04-10",
                    "check_out": "2026-04-13",
                    "budget": 200,
                    "limit": 2,
                },
            )
            print("\nHotel results:")
            print(hotel_result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
