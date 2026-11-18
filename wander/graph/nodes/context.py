from dataclasses import dataclass
from typing import Any


@dataclass
class TripNodeContext:
    model: Any
    coordinator_model: Any
    validator_model: Any
    intent_model: Any
    schemas: dict[str, dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    activities_graph: Any
    accommodation_graph: Any
