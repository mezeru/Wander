from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_FILE = Path(__file__).resolve().parent / "data" / "destinations.json"


def _load_destination_index() -> dict[str, Any]:
    with _DATA_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_destination_name(destination: str) -> str:
    return destination.strip().title()


def get_destination_info(destination: str) -> dict[str, Any]:
    destination_name = _normalize_destination_name(destination)
    data = _load_destination_index()
    if destination_name not in data:
        raise KeyError(f"Destination '{destination}' not found in mock travel data.")
    return data[destination_name]


def get_attractions(destination: str, interests: list[str] | None = None, limit: int = 5) -> list[dict[str, Any]]:
    destination_data = get_destination_info(destination)
    attractions = destination_data["attractions"]
    requested_tags = {tag.strip().lower() for tag in (interests or [])}

    ranked: list[tuple[int, dict[str, Any]]] = []
    for attraction in attractions:
        tags = {str(tag).lower() for tag in attraction.get("tags", [])}
        match_score = len(requested_tags & tags) if requested_tags else 1
        ranked.append((match_score, attraction))

    ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
    results = [attraction for _, attraction in ranked[:limit]]
    for attraction in results:
        attraction["match_score"] = max(
            1,
            len({str(tag).lower() for tag in attraction.get("tags", [])} & requested_tags),
        ) if requested_tags else 1
    return results


def search_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    budget: float,
    limit: int = 5,
) -> list[dict[str, Any]]:
    destination_data = get_destination_info(destination)
    hotels = destination_data["hotels"]
    results: list[dict[str, Any]] = []
    for hotel in hotels:
        nightly_rate = float(hotel["nightly_rate"])
        if nightly_rate <= float(budget):
            results.append(
                {
                    "name": hotel["name"],
                    "area": hotel["area"],
                    "nightly_rate": nightly_rate,
                    "amenities": hotel.get("amenities", []),
                    "check_in": check_in,
                    "check_out": check_out,
                    "typical_trip_match": hotel.get("typical_trip_match", ""),
                }
            )
    results.sort(key=lambda item: item["nightly_rate"])
    return results[:limit]


def get_travel_times(origin: str, destination: str, transport_mode: str) -> dict[str, Any]:
    origin_name = origin.strip()
    destination_name = destination.strip()
    normalized_mode = transport_mode.strip().lower()

    base_estimates = {
        "flight": {"duration_min": 240, "cost_estimate": 430},
        "train": {"duration_min": 180, "cost_estimate": 90},
        "car": {"duration_min": 320, "cost_estimate": 120},
    }

    route = base_estimates.get(normalized_mode, {"duration_min": 180, "cost_estimate": 100})
    return {
        "origin": origin_name,
        "destination": destination_name,
        "transport_mode": normalized_mode,
        "duration_min": route["duration_min"],
        "cost_estimate": route["cost_estimate"],
        "notes": "Illustrative travel time based on mock data; validate against live APIs before booking.",
    }
