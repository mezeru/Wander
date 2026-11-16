import asyncio

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import AnyUrl

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "wander.mcp_server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # 1. Initialize MCP session
            await session.initialize()

            # Discover resource templates
            prompts = session.list_prompts()

            print(prompts)

            prompt = session.get_prompt("trip_planning_prompt", arguments={"destination":"Kyoto","days": 5})


if __name__ == "__main__":
    asyncio.run(main())