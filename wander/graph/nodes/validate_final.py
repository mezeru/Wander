import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..schemas import TripState
from .context import TripNodeContext


def create_validate_final_node(context: TripNodeContext):
    async def validate_final(state: TripState) -> dict[str, Any]:
        coverage = state.get("research_coverage", {})
        incomplete = bool(
            coverage.get("failed")
            or coverage.get("no_data")
            or coverage.get("missing")
            or coverage.get("duplicate")
            or coverage.get("unknown")
        )
        result = await context.validator_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Review the itinerary against the original user request AND research evidence. "
                        "Flag unsupported or invented attractions, hotels, prices, availability, inaccurate "
                        "currencies, duration inconsistencies, and budget violations. "
                        "Account for research coverage explicitly supplied in the prompt. "
                        "Do not mark a partial or integrity-flagged itinerary fully verified."
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
                        f"\nResearch coverage:\n{json.dumps(coverage, sort_keys=True)}"
                    )
                ),
            ]
        )
        if incomplete:
            issues = list(result.issues)
            issues.append(
                "Research coverage is incomplete; missing coverage must remain explicit "
                "and the itinerary must not be marked fully verified."
            )
            return {"issues": issues, "status": "partial"}
        return {"issues": result.issues, "status": result.status}

    return validate_final
