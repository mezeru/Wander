import json

from ..schemas import TripState
from .context import TripNodeContext


def create_accommodation_node(context: TripNodeContext):
    async def accommodation(state: TripState) -> dict[str, str]:
        args = state["tool_arguments"]["search_hotels"]
        response = await context.accommodation_graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{state['accommodation_task']}\n\n"
                            "Use these validated hotel-search arguments: "
                            f"{json.dumps(args)}. "
                            "These hotels are mock suggestions, not confirmed availability."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 15},
        )
        return {
            "research_results": [{
                "job_id": state["job_id"],
                "agent": state["agent"],
                "destination": state["destination"],
                "status": "completed",
                "result": response["messages"][-1].content,
            }]
        }

    return accommodation
