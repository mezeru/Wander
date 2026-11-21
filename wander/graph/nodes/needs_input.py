from ..schemas import TripState


def needs_input(_state: TripState) -> dict[str, str]:
    return {"status": "needs_input", "status_message": "Clarification limit reached. The listed details are still required."}
