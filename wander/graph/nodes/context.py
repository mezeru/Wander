from dataclasses import dataclass
from typing import Any


@dataclass
class TripNodeContext:
    model: Any = None
    intent_model: Any = None
    schemas: dict[str, dict[str, Any]] | None = None
    live_providers: bool = False
    activities_graph: Any = None
    accommodation_graph: Any = None
