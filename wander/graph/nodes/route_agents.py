from typing import Any

from ..schemas import TripState


def route_agents(_: TripState) -> dict[str, Any]:
    return {}


def ready_for_dispatch(_: TripState) -> dict[str, str]:
    return {"status": "ready_for_dispatch"}
