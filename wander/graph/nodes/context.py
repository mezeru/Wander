from dataclasses import dataclass
from typing import Any


@dataclass
class TripNodeContext:
    model: Any = None
    coordinator_model: Any = None
    validator_model: Any = None
    intent_model: Any = None
    schemas: dict[str, dict[str, Any]] | None = None
    tool_specs: list[dict[str, Any]] | None = None
    activities_graph: Any = None
    accommodation_graph: Any = None
