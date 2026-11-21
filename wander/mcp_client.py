import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from wander.mcp_runtime import server_parameters


async def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    server_params = server_parameters(project_root)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            prompts = await session.list_prompts()
            print(prompts.model_dump())
            prompt = await session.get_prompt(
                "trip_planning_prompt",
                arguments={"destinations": "Kyoto", "days": 5},
            )
            print(prompt.model_dump())


if __name__ == "__main__":
    asyncio.run(main())