from ..schemas import TripState


def needs_input(state: TripState) -> dict[str, str]:
    if state.get("research_coverage"):
        return {"status": "research_failed"}
    return {"status": "needs_input"}
