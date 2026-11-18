import logging
from typing import Any

from ..schemas import ResearchWorkerState
from .context import TripNodeContext


logger = logging.getLogger(__name__)


def create_research_worker_node(context: TripNodeContext):
    async def research_worker(
        state: ResearchWorkerState,
    ) -> dict[str, list[dict[str, Any]]]:
        agent_name = state["agent"]
        logger.info("Research worker selected specialist: %s", agent_name)

        if agent_name == "activities":
            graph = context.activities_graph
        elif agent_name == "accommodation":
            graph = context.accommodation_graph
        else:
            raise ValueError(f"Unknown research agent: {agent_name}")

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{state['task']}\n\n"
                            "Research arguments:\n"
                            f"{state['arguments']}"
                        ),
                    }
                ]
            }
        )
        return {
            "research_results": [
                {
                    "agent": agent_name,
                    "result": result["messages"][-1].content,
                }
            ]
        }

    return research_worker
