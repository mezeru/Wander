from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import httpx

from ..graph.mcp_schemas import Hotel
from .config import (
    AmbiguousDestinationError,
    DestinationNotFoundError,
    TravelConfigurationError,
    required_api_key,
    travel_settings,
)
from .http import get_json


logger = logging.getLogger(__name__)
_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
_PLACES_URL = "https://api.geoapify.com/v2/places"
_LODGING_CATEGORY = "accommodation"
_PREFERRED_CATEGORIES = "accommodation.hotel,accommodation.hostel,accommodation.guest_house"
_ACCOMMODATION_INTERESTS = {
    "hotel": "accommodation.hotel",
    "hotels": "accommodation.hotel",
    "hostel": "accommodation.hostel",
    "hostels": "accommodation.hostel",
    "guest house": "accommodation.guest_house",
    "guesthouse": "accommodation.guest_house",
    "guesthouses": "accommodation.guest_house",
    "apartment": "accommodation.apartment",
    "apartments": "accommodation.apartment",
    "motel": "accommodation.motel",
    "motels": "accommodation.motel",
    "chalet": "accommodation.chalet",
    "chalets": "accommodation.chalet",
    "hut": "accommodation.hut",
    "huts": "accommodation.hut",
}


def _categories_for_interests(interests: list[str] | None) -> str:
    """Prefer common lodging types unless the caller explicitly selects types."""
    if not interests:
        return _PREFERRED_CATEGORIES
    if not isinstance(interests, list):
        raise TravelConfigurationError("Accommodation interests must be a list of strings.")
    categories: set[str] = set()
    unsupported: list[str] = []
    for interest in interests:
        if not isinstance(interest, str) or not interest.strip():
            raise TravelConfigurationError("Each accommodation interest must be non-empty.")
        value = interest.strip().casefold()
        if value in ("all", "any", "accommodation", "lodging"):
            return _LODGING_CATEGORY
        category = _ACCOMMODATION_INTERESTS.get(value)
        if category is None:
            unsupported.append(interest)
        else:
            categories.add(category)
    if unsupported:
        raise TravelConfigurationError(
            "Unsupported accommodation type filters: " + ", ".join(unsupported)
            + ". Geoapify cannot filter by price, luxury, or amenities here; "
            "supported types: " + ", ".join(sorted(_ACCOMMODATION_INTERESTS))
        )
    return ",".join(sorted(categories))


def _normalize_name(value: Any) -> str:
    return " ".join(value.strip().casefold().split()) if isinstance(value, str) else ""


def _coordinates(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) < 2
        or any(isinstance(item, bool) for item in value[:2])
    ):
        return None
    try:
        longitude, latitude = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90 <= latitude <= 90
        or not -180 <= longitude <= 180
    ):
        return None
    return latitude, longitude


def _feature_coordinates(
    feature: dict[str, Any], properties: dict[str, Any]
) -> tuple[float, float] | None:
    geometry = feature.get("geometry")
    if isinstance(geometry, dict) and geometry.get("type", "Point") == "Point":
        coordinates = _coordinates(geometry.get("coordinates"))
        if coordinates is not None:
            return coordinates
    return _coordinates((properties.get("lon"), properties.get("lat")))


async def _resolve_destination(
    client: httpx.AsyncClient,
    destination: str,
    api_key: str,
) -> tuple[float, float]:
    logger.info("Geoapify: resolving destination %r", destination)
    payload = await get_json(
        client,
        _GEOCODE_URL,
        {
            "text": destination,
            "type": "city",
            "format": "geojson",
            "apiKey": api_key,
        },
        "Geoapify",
    )
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise DestinationNotFoundError(
            f"Geoapify could not resolve destination '{destination}'."
        )
    logger.info("Geoapify: destination=%r geocoding_candidates=%d", destination, len(features))
    requested_name = _normalize_name(destination)
    matches: set[tuple[str, tuple[float, float]]] = set()
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        components = properties.get("datasource")
        components = components.get("raw") if isinstance(components, dict) else None
        city_names = {
            _normalize_name(properties.get("city")),
            _normalize_name(properties.get("name")),
            _normalize_name(components.get("city")) if isinstance(components, dict) else "",
        }
        if requested_name not in city_names:
            continue
        coords = _feature_coordinates(feature, properties)
        if coords is None:
            continue
        country = _normalize_name(properties.get("country"))
        if not country and isinstance(components, dict):
            country = _normalize_name(components.get("country"))
        matches.add((country, coords))

    if not matches:
        raise DestinationNotFoundError(
            f"Geoapify could not find an exact city match for '{destination}'."
        )
    if len(matches) > 1:
        raise AmbiguousDestinationError(
            f"Geoapify returned multiple city matches for '{destination}'; provide a more specific destination."
        )
    return next(iter(matches))[1]


def _normalize_hotel(
    feature: Any,
    destination: str,
) -> Hotel | str:
    """Return a validated hotel or the first reason the feature was rejected."""
    properties = feature.get("properties") if isinstance(feature, dict) else None
    if not isinstance(properties, dict):
        return "malformed"
    place_id = properties.get("place_id")
    name = properties.get("name")
    if not isinstance(name, str) or not name.strip():
        return "missing_name"
    if not isinstance(place_id, str) or not place_id.strip():
        return "missing_place_id"
    coordinates = _feature_coordinates(feature, properties)
    if coordinates is None:
        return "invalid_geometry"
    latitude, longitude = coordinates

    categories = properties.get("categories")
    labels = (
        [item for item in categories if isinstance(item, str) and item.strip()]
        if isinstance(categories, list) else []
    )
    if labels and not any(
        label == _LODGING_CATEGORY or label.startswith(f"{_LODGING_CATEGORY}.")
        for label in labels
    ):
        return "non_accommodation"
    address = properties.get("formatted")
    if not isinstance(address, str) or not address.strip():
        address = properties.get("address_line1")
    website = properties.get("website")
    return Hotel(
        id=place_id.strip(),
        name=name.strip(),
        nightly_price=None,
        availability_verified=False,
        source="Geoapify",
        destination=destination,
        provider="geoapify",
        category=", ".join(labels) or None,
        address=address if isinstance(address, str) else None,
        coordinates={"latitude": latitude, "longitude": longitude},
        source_url=website if isinstance(website, str) else None,
        website=website if isinstance(website, str) else None,
        pricing_available=False,
        provider_metadata={
            key: properties[key]
            for key in (
                "datasource",
                "datasource_name",
                "rank",
                "place_type",
                "result_type",
            )
            if key in properties
        },
    )


async def _search(
    client: httpx.AsyncClient,
    destination: str,
    check_in: str,
    check_out: str,
    budget: float,
    limit: int,
    api_key: str,
    interests: list[str] | None = None,
) -> list[dict[str, Any]]:
    settings = travel_settings()
    target = min(limit, settings.max_results)
    categories = _categories_for_interests(interests)
    latitude, longitude = await _resolve_destination(client, destination, api_key)
    candidate_limit = settings.candidate_limit(target)
    logger.info(
        "Geoapify: destination=%r selected lat=%s lon=%s requested=%d target=%d candidate_limit=%d",
        destination, latitude, longitude, limit, target, candidate_limit,
    )

    params = {
        "bias": f"proximity:{longitude},{latitude}",
        "limit": candidate_limit,
        "apiKey": api_key,
    }
    primary = (categories, settings.search_radius_meters)
    fallback = (
        categories if interests else _LODGING_CATEGORY,
        max(settings.search_radius_meters, settings.fallback_radius_meters),
    )
    hotels: dict[str, Hotel] = {}
    totals: Counter[str] = Counter()
    fallback_used = False
    for index, (category, radius) in enumerate(dict.fromkeys((primary, fallback))):
        fallback_used = index > 0
        search_params = {
            **params,
            "categories": category,
            "filter": f"circle:{longitude},{latitude},{radius}",
        }
        payload = await get_json(client, _PLACES_URL, search_params, "Geoapify")
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            raise TravelConfigurationError(
                "Geoapify returned an unexpected accommodation-search response."
            )
        counts: Counter[str] = Counter(raw=len(features))
        for feature in features:
            hotel = _normalize_hotel(feature, destination)
            if isinstance(hotel, str):
                counts[hotel] += 1
                continue
            counts["normalized"] += 1
            if hotel.id in hotels:
                counts["duplicates"] += 1
                continue
            hotels[hotel.id] = hotel
        totals.update(counts)
        logger.info(
            "Geoapify: destination=%r radius_m=%d categories=%s fallback=%s raw=%d "
            "missing_name=%d missing_place_id=%d invalid_geometry=%d malformed=%d "
            "non_accommodation=%d normalized=%d duplicates=%d",
            destination, radius, category, fallback_used, counts["raw"], counts["missing_name"],
            counts["missing_place_id"], counts["invalid_geometry"], counts["malformed"],
            counts["non_accommodation"], counts["normalized"], counts["duplicates"],
        )
        if len(hotels) >= target:
            break

    results = [
        {
            **hotel.model_dump(mode="json"),
            "check_in": check_in,
            "check_out": check_out,
            "date_availability_checked": False,
            "budget_constraint": budget,
            "pricing_note": (
                "Geoapify provides place discovery only; room pricing and "
                "availability were not checked."
            ),
        }
        for hotel in list(hotels.values())[:target]
    ]
    logger.info(
        "Geoapify: destination=%r raw=%d rejected=%d normalized=%d duplicates=%d "
        "valid=%d fallback=%s returned=%d",
        destination, totals["raw"], totals["raw"] - totals["normalized"],
        totals["normalized"], totals["duplicates"], len(hotels), fallback_used, len(results),
    )
    return results


async def search_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    budget: float,
    limit: int = 5,
    client: httpx.AsyncClient | None = None,
    interests: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(destination, str) or not destination.strip():
        raise DestinationNotFoundError("Destination must be a non-empty city name.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise TravelConfigurationError("Accommodation result limit must be positive.")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)):
        raise TravelConfigurationError("Hotel budget must be numeric.")
    _categories_for_interests(interests)  # Validate before any requests.
    settings = travel_settings()
    api_key = required_api_key("GEOAPIFY_API_KEY", "Geoapify")
    if client is not None:
        return await _search(
            client, destination.strip(), check_in, check_out, budget, limit,
            api_key, interests,
        )
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as owned_client:
        return await _search(
            owned_client, destination.strip(), check_in, check_out, budget, limit,
            api_key, interests,
        )
