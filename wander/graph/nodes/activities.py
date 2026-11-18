import json

from ..schemas import TripState
from .context import TripNodeContext


def create_activities_node(context: TripNodeContext):
    async def activities(state: TripState) -> dict[str, str]:
        args = state["tool_arguments"]["search_attractions"]
        response = await context.activities_graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{state['activities_task']}\n\n"
                            "Use these validated attraction-search arguments: "
                            f"{json.dumps(args)}. "
                            "Do not invent tool inputs or claim mock data is live."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 15},
        )
        return {"activities_result": str(response["messages"][-1].content)}

    return activities
