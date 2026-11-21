from typing import Any

from ..schemas import TripState


def ready_for_dispatch(_: TripState) -> dict[str, Any]:
    return {}
