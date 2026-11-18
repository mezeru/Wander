import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState
from .context import TripNodeContext


def create_validate_final_node(context: TripNodeContext):
    async def validate_final(state: TripState) -> dict[str, Any]:
        result = await context.validator_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Review the itinerary against the original user request AND research evidence. "
                        "Flag unsupported or invented attractions, hotels, prices, availability, inaccurate "
                        "currencies, duration inconsistencies, and budget violations. "
                        "Don't confuse nightly hotel budget with total trip budget. "
                        "Return 'verified' only if no issues are found relative to the supplied mock data; "
                        "this does not mean hotels or prices are verified in the real world."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Request:\n{state['user_request']}\n\n"
                        f"Validated arguments:\n{json.dumps(state['tool_arguments'])}\n\n"
                        f'''Research results:
                        {json.dumps(state["research_results"], indent=2)}
                    '''
                        f"Final plan:\n{state['final_plan']}"
                    )
                ),
            ]
        )
        return {"issues": result.issues, "status": result.status}

    return validate_final
