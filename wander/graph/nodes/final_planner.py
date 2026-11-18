import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState
from .context import TripNodeContext


def create_final_planner_node(context: TripNodeContext):
    async def final_planner(state: TripState) -> dict[str, Any]:
        messages = [
            SystemMessage(
                content=(
                    "You are the WanderMind itinerary planner. Combine only the supplied research. "
                    "Preserve the user's constraints and validated arguments; don't assume number of nights "
                    "from the number of days. All travel data is illustrative, not verified live availability. "
                    "Do not invent hotel names, costs, attractions, exchange rates, or transport times. "
                    "If information is missing, describe it instead of inventing it."
                )
            ),
            HumanMessage(
                content=(
                    f"Original request:\n{state['user_request']}\n\n"
                    f"Validated arguments:\n{json.dumps(state['tool_arguments'])}\n\n"
                    f'''Research results:
                        {json.dumps(state["research_results"], indent=2)}
                    '''
                )
            ),
        ]
        if state.get("issues"):
            messages.append(
                HumanMessage(
                    content=(
                        f"Previous plan:\n{state.get('final_plan', '')}\n\n"
                        "Revise to address these validation issues:\n"
                        f"{json.dumps(state['issues'])}"
                    )
                )
            )
        response = await context.model.ainvoke(messages)
        return {
            "final_plan": str(response.content),
            "plan_attempts": state.get("plan_attempts", 0) + 1,
        }

    return final_planner
