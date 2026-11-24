from ..schemas import TripState


def needs_input(state: TripState) -> dict[str, str]:
    coverage = state.get("research_coverage")
    trip_intent = state.get("trip_intent", {})
    if not isinstance(trip_intent, dict):
        dump = getattr(trip_intent, "model_dump", None)
        trip_intent = dump() if callable(dump) else {}
    if (
        isinstance(coverage, dict)
        and coverage.get("completed", 0) > 0
        and trip_intent.get("needs_activities") is False
    ):
        has_limitations = any(
            coverage.get(field)
            for field in ("failed", "no_data", "missing", "duplicate", "unknown")
        )
        return {"status": "partial" if has_limitations else "verified"}
    if coverage:
        return {"status": "research_failed"}
    return {"status": "needs_input"}
