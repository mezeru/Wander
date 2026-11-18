from ..schemas import TripState


def needs_input(_: TripState) -> dict[str, str]:
    return {"status": "needs_input"}
