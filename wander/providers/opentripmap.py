from __future__ import annotations

import logging
import math
from asyncio import sleep
from collections.abc import Callable, Iterator
from threading import Lock
from time import monotonic
from typing import Any

import httpx

from ..graph.mcp_schemas import Attraction, Coordinates
from .config import (
    AmbiguousDestinationError,
    DestinationNotFoundError,
    TravelConfigurationError,
    required_api_key,
    travel_settings,
)
from .http import get_json


logger = logging.getLogger(__name__)
_BASE_URL = "https://api.opentripmap.com/0.1/en/places"
_EARTH_RADIUS_METERS = 6_371_000
_REQUEST_INTERVAL_SECONDS = 0.25
_request_slot_lock = Lock()
_next_request_at = 0.0
# User-friendly interest names -> OpenTripMap category keys.
# An empty/omitted interest list means NO kinds filter (all tourist categories).
_KIND_FILTERS = {
    "architecture": "architecture",
    "art": "cultural",
    "arts": "cultural",
    "culture": "cultural",
    "cultural": "cultural",
    "history": "historic",
    "historic": "historic",
    "heritage": "historic",
    "museums": "museums",
    "museum": "museums",
    "nature": "natural",
    "natural": "natural",
    "outdoors": "natural",
    "parks": "gardens_and_parks",
    "gardens": "gardens_and_parks",
    "religion": "religion",
    "religious": "religion",
    "temples": "religion",
    "shrines": "religion",
    "beaches": "beaches",
    "beach": "beaches",
    "food": "foods",
    "restaurants": "foods",
    "shopping": "shops",
    "shops": "shops",
    "entertainment": "amusements",
    "amusement": "amusements",
    "sports": "sport",
    "sightseeing": "interesting_places",
    "landmarks": "interesting_places",
}
_SIGHTSEEING_KINDS = {
    "architecture", "gardens_and_parks", "historic", "museums", "natural", "religion",
}


async def _pace_request(_request: httpx.Request) -> None:
    """Space live requests from the same process to avoid provider bursts."""
    global _next_request_at
    with _request_slot_lock:
        now = monotonic()
        slot = max(now, _next_request_at)
        _next_request_at = slot + _REQUEST_INTERVAL_SECONDS
    if slot > now:
        await sleep(slot - now)


def _coordinate(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise DestinationNotFoundError(
            f"OpenTripMap could not resolve valid coordinates for the destination ({label})."
        )
    try:
        coordinate = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DestinationNotFoundError(
            f"OpenTripMap could not resolve valid coordinates for the destination ({label})."
        ) from None
    if not math.isfinite(coordinate) or not minimum <= coordinate <= maximum:
        raise DestinationNotFoundError(
            f"OpenTripMap could not resolve valid coordinates for the destination ({label})."
        )
    return coordinate


async def _resolve_destination(
    client: httpx.AsyncClient,
    destination: str,
    api_key: str,
) -> tuple[float, float]:
    logger.info("OpenTripMap: resolving destination %r", destination)
    payload = await get_json(
        client,
        f"{_BASE_URL}/geoname",
        {"name": destination, "apikey": api_key},
        "OpenTripMap",
    )
    if isinstance(payload, list):
        candidates = [
            item for item in payload
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and item["name"].strip().casefold() == destination.strip().casefold()
        ]
        if not candidates:
            raise DestinationNotFoundError(
                f"OpenTripMap could not find destination '{destination}'."
            )
        if len(candidates) != 1:
            raise AmbiguousDestinationError(
                f"OpenTripMap returned multiple matches for '{destination}'."
            )
        payload = candidates[0]
    if not isinstance(payload, dict):
        raise DestinationNotFoundError(
            f"OpenTripMap could not find destination '{destination}'."
        )

    resolved_name = payload.get("name")
    if (
        not isinstance(resolved_name, str)
        or resolved_name.strip().casefold() != destination.strip().casefold()
    ):
        raise DestinationNotFoundError(
            f"OpenTripMap did not resolve an exact city match for '{destination}'."
        )
    latitude = _coordinate(payload.get("lat"), "latitude", -90, 90)
    longitude = _coordinate(payload.get("lon"), "longitude", -180, 180)
    return latitude, longitude


def _kinds_for_interests(interests: list[str] | None) -> str | None:
    """Apply a category filter only when the caller supplied interests.

    Reject unknown interests rather than silently returning unfiltered results.
    """
    if not interests:
        return None
    if not isinstance(interests, list):
        raise TravelConfigurationError("Interests must be a list of strings.")
    resolved: set[str] = set()
    unsupported: list[str] = []
    for interest in interests:
        if not isinstance(interest, str) or not interest.strip():
            raise TravelConfigurationError("Each interest must be a non-empty string.")
        key = interest.strip().casefold()
        category = _KIND_FILTERS.get(key)
        if category is None:
            unsupported.append(interest)
        else:
            resolved.add(category)
    if unsupported:
        raise TravelConfigurationError(
            "Unsupported attraction interests: " + ", ".join(unsupported)
            + ". Supported interests: " + ", ".join(sorted(_KIND_FILTERS))
        )
    return ",".join(sorted(resolved)) or None


def _popularity(value: Any) -> float:
    """OpenTripMap can return ratings as numbers or strings like '2'/'2h'."""
    if isinstance(value, bool):
        return 0.0
    try:
        rating = float(str(value).strip().lower().removesuffix("h"))
    except (ValueError, TypeError):
        return 0.0
    return rating if math.isfinite(rating) and rating > 0 else 0.0


def _distance(value: Any) -> float:
    try:
        distance = float(value)
    except (ValueError, TypeError, OverflowError):
        return math.inf
    return distance if math.isfinite(distance) and distance >= 0 else math.inf


def _summary_coordinates(summary: dict[str, Any]) -> Coordinates | None:
    point = summary.get("point")
    if not isinstance(point, dict):
        return None
    try:
        return Coordinates(
            latitude=_coordinate(point.get("lat"), "latitude", -90, 90),
            longitude=_coordinate(point.get("lon"), "longitude", -180, 180),
        )
    except DestinationNotFoundError:
        return None


def _spread[T](
    ranked: list[T],
    coordinates: Callable[[T], Coordinates | None],
    spacing: int,
) -> tuple[list[T], list[T]]:
    if spacing == 0:
        return ranked, []
    preferred: list[T] = []
    nearby: list[T] = []
    points: list[tuple[float, float]] = []
    # Compare haversines directly using the mean Earth radius in metres.
    threshold = math.sin(min(spacing / (2 * _EARTH_RADIUS_METERS), math.pi / 2)) ** 2
    for item in ranked:
        point = coordinates(item)
        if point is not None:
            lat, lon = math.radians(point.latitude), math.radians(point.longitude)
            if any(
                _haversine(lat, lon, other_lat, other_lon) < threshold
                for other_lat, other_lon in points
            ):
                nearby.append(item)
                continue
            points.append((lat, lon))
        preferred.append(item)
    return preferred, nearby


def _haversine(lat: float, lon: float, other_lat: float, other_lon: float) -> float:
    return (
        math.sin((lat - other_lat) / 2) ** 2
        + math.cos(lat) * math.cos(other_lat) * math.sin((lon - other_lon) / 2) ** 2
    )


def _outer_centers(latitude: float, longitude: float, distance: float) -> Iterator[tuple[float, float]]:
    lat, lon = math.radians(latitude), math.radians(longitude)
    arc = distance / _EARTH_RADIUS_METERS
    for bearing in (0, math.pi / 2, math.pi, 3 * math.pi / 2):
        outer_lat = math.asin(
            math.sin(lat) * math.cos(arc) + math.cos(lat) * math.sin(arc) * math.cos(bearing)
        )
        outer_lon = lon + math.atan2(
            math.sin(bearing) * math.sin(arc) * math.cos(lat),
            math.cos(arc) - math.sin(lat) * math.sin(outer_lat),
        )
        yield math.degrees(outer_lat), (math.degrees(outer_lon) + 180) % 360 - 180


def _summary_rank(summary: dict[str, Any]) -> tuple[int, float, int, int, float, str]:
    kinds = summary.get("kinds")
    categories = {kind.strip() for kind in kinds.split(",")} if isinstance(kinds, str) else set()
    return (
        int("accomodations" in categories),  # OpenTripMap's category spelling.
        -_popularity(summary.get("rate")),
        -int(bool(summary.get("wikidata"))),
        -int(bool(categories & _SIGHTSEEING_KINDS)),
        _distance(summary.get("dist")),
        summary["xid"],
    )


def _detail_rank(
    attraction: Attraction, summary: dict[str, Any]
) -> tuple[int, float, int, int, int, float, str]:
    lodging, rate, wikidata, sightseeing, distance, xid = _summary_rank(summary)
    metadata = attraction.provider_metadata
    detail_rate = _popularity(metadata.get("rate"))
    return (
        lodging,
        -detail_rate if detail_rate else rate,
        -int(bool(attraction.description)),
        -1 if metadata.get("wikipedia") else wikidata,
        sightseeing,
        distance,
        xid,
    )


def _description(detail: dict[str, Any]) -> str | None:
    extracts = detail.get("wikipedia_extracts")
    if isinstance(extracts, dict) and isinstance(extracts.get("text"), str):
        text = extracts["text"].strip()
        if text:
            return text
    info = detail.get("info")
    if isinstance(info, dict) and isinstance(info.get("descr"), str):
        text = info["descr"].strip()
        if text:
            return text
    return None


def _normalize_attraction(
    detail: dict[str, Any],
    summary: dict[str, Any],
    destination: str,
    xid: str,
) -> Attraction | None:
    name = detail.get("name")
    if not isinstance(name, str) or not name.strip():
        name = summary.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    kinds = detail.get("kinds") or summary.get("kinds")
    category = (
        ", ".join(
            part.strip().replace("_", " ")
            for part in kinds.split(",")
            if part.strip()
        )
        if isinstance(kinds, str)
        else "attraction"
    )
    coordinates = _summary_coordinates(detail) or _summary_coordinates(summary)

    source_url = detail.get("url") or detail.get("otm")
    if not isinstance(source_url, str) or not source_url.strip():
        source_url = None
    address = detail.get("address")
    if not isinstance(address, (dict, str)):
        address = None
    website = detail.get("website")
    if not isinstance(website, str) or not website.strip():
        website = None
    metadata = {
        key: detail[key]
        for key in (
            "rate",
            "kinds",
            "image",
            "wikipedia",
            "preview",
            "info",
            "wikipedia_extracts",
        )
        if key in detail
    }
    if "rate" not in metadata and "rate" in summary:
        metadata["rate"] = summary["rate"]
    if "dist" in summary:
        metadata["distance_from_center_meters"] = summary["dist"]
    return Attraction(
        id=xid,
        name=name.strip(),
        category=category or "attraction",
        estimated_duration_hours=None,
        price=None,
        source="OpenTripMap",
        verified_live=False,
        destination=destination,
        provider="opentripmap",
        coordinates=coordinates,
        description=_description(detail),
        source_url=source_url,
        address=address,
        website=website,
        provider_metadata=metadata,
    )


async def _search(
    client: httpx.AsyncClient,
    destination: str,
    interests: list[str] | None,
    limit: int,
    api_key: str,
) -> list[dict[str, Any]]:
    settings = travel_settings()
    target = min(limit, settings.max_results)
    kinds = _kinds_for_interests(interests)  # Validate before making requests.
    latitude, longitude = await _resolve_destination(client, destination, api_key)
    candidate_limit = settings.candidate_limit(target)
    spacing = settings.opentripmap_min_spacing_meters
    logger.info(
        "OpenTripMap: destination=%r selected lat=%s lon=%s requested=%d target=%d candidate_limit=%d",
        destination, latitude, longitude, limit, target, candidate_limit,
    )

    params = {
        "format": "json",
        "apikey": api_key,
    }
    if kinds is not None:
        params["kinds"] = kinds
    radius = settings.search_radius_meters
    expanded_radius = max(radius, settings.fallback_radius_meters)
    min_rate = settings.opentripmap_min_rate
    searches = list(dict.fromkeys((
        (radius, min_rate),
        (expanded_radius, min_rate),
        (expanded_radius, 1),
    )))

    summaries: dict[str, dict[str, Any]] = {}
    attractions: list[Attraction] = []
    attempted_ids: set[str] = set()
    preferred: list[Attraction] = []
    nearby: list[Attraction] = []
    raw_count = rejected = duplicates = detail_rejections = 0
    fallback_used = False
    sampled_outer_areas = False
    center_lat, center_lon = math.radians(latitude), math.radians(longitude)

    async def discover(
        lat: float, lon: float, search_radius: int, rate: int, batch_limit: int, *, outer: bool,
    ) -> list[Any]:
        nonlocal raw_count, rejected, duplicates
        records = await get_json(
            client, f"{_BASE_URL}/radius",
            {**params, "lat": lat, "lon": lon, "radius": search_radius,
             "rate": str(rate), "limit": batch_limit},
            "OpenTripMap",
        )
        if not isinstance(records, list):
            raise TravelConfigurationError(
                "OpenTripMap returned an unexpected attraction-search response."
            )

        raw_count += len(records)
        added = 0
        for record in records:
            if not isinstance(record, dict):
                rejected += 1
                continue
            xid, name = record.get("xid"), record.get("name")
            if (
                not isinstance(xid, str) or not xid.strip()
                or not isinstance(name, str) or not name.strip()
            ):
                rejected += 1
                continue
            xid = xid.strip()
            if xid in summaries:
                duplicates += 1
                continue
            summary = {**record, "xid": xid}
            if outer:
                # Radius results measure distance from the queried point.
                summary.pop("dist", None)
                point = _summary_coordinates(summary)
                if point is not None:
                    distance = _haversine(
                        center_lat, center_lon, math.radians(point.latitude), math.radians(point.longitude),
                    )
                    summary["dist"] = 2 * _EARTH_RADIUS_METERS * math.asin(math.sqrt(min(1.0, distance)))
            summaries[xid] = summary
            added += 1
        logger.info(
            "OpenTripMap: destination=%r radius_m=%d min_rate=%d candidate_limit=%d "
            "fallback=%s outer_area=%s raw=%d new_candidates=%d",
            destination, search_radius, rate, batch_limit, fallback_used, outer, len(records), added,
        )
        return records

    for index, (search_radius, rate) in enumerate(searches):
        fallback_used |= index > 0
        last_search = index == len(searches) - 1
        records = await discover(latitude, longitude, search_radius, rate, candidate_limit, outer=False)
        offset = search_radius // 4
        # A full nearest-first batch can stop well short of the configured radius.
        # Sample four additional centers once, with each circle inside that radius.
        if (
            spacing > 0 and not sampled_outer_areas and offset >= 1000
            and len(records) >= candidate_limit
            and max(_distance(record.get("dist")) if isinstance(record, dict) else math.inf
                    for record in records) < offset
        ):
            fallback_used = sampled_outer_areas = True
            for outer_lat, outer_lon in _outer_centers(latitude, longitude, offset):
                await discover(
                    outer_lat, outer_lon, search_radius // 2, rate,
                    min(candidate_limit, 24), outer=True,
                )

        candidates, clustered = _spread(
            sorted(summaries.values(), key=_summary_rank), _summary_coordinates, spacing,
        )
        if len(candidates) < target and not last_search:
            continue
        if last_search:
            candidates += clustered

        # Reserve up to a third of the detail budget for each discovery pass.
        # One extra usable detail lets metadata break a tie in the shortlist.
        detail_limit = (index + 1) * (target + 1) if not last_search else 3 * target + 2
        shortlist_complete = False
        for summary in candidates:
            if len(attempted_ids) >= detail_limit:
                break
            xid = summary["xid"]
            if xid in attempted_ids:
                continue
            attempted_ids.add(xid)
            detail = await get_json(
                client, f"{_BASE_URL}/xid/{xid}", {"apikey": api_key}, "OpenTripMap",
            )
            attraction = (
                _normalize_attraction(detail, summary, destination, xid)
                if isinstance(detail, dict) else None
            )
            if attraction is None:
                detail_rejections += 1
                continue
            attractions.append(attraction)
            ranked = sorted(
                attractions, key=lambda item: _detail_rank(item, summaries[item.id]),
            )
            preferred, nearby = _spread(ranked, lambda item: item.coordinates, spacing)
            if len(preferred) >= target:
                if shortlist_complete:
                    break
                shortlist_complete = True
        if len(preferred) >= target:
            break

    results = (preferred + nearby)[:target]
    logger.info(
        "OpenTripMap: destination=%r raw=%d candidate_pool=%d rejected=%d duplicates=%d "
        "detail_requests=%d detail_rejected=%d valid=%d nearby_deferred=%d "
        "fallback=%s sampled_outer_areas=%s returned=%d",
        destination, raw_count, len(summaries), rejected, duplicates, len(attempted_ids),
        detail_rejections, len(attractions), len(nearby), fallback_used, sampled_outer_areas, len(results),
    )
    return [item.model_dump(mode="json") for item in results]


async def search_attractions(
    destination: str,
    interests: list[str] | None = None,
    limit: int = 5,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(destination, str) or not destination.strip():
        raise DestinationNotFoundError("Destination must be a non-empty city name.")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise TravelConfigurationError("Attraction result limit must be positive.")
    settings = travel_settings()
    api_key = required_api_key("OPENTRIPMAP_API_KEY", "OpenTripMap")
    if client is not None:
        return await _search(client, destination.strip(), interests, limit, api_key)
    async with httpx.AsyncClient(
        timeout=settings.timeout_seconds, event_hooks={"request": [_pace_request]},
    ) as owned_client:
        return await _search(
            owned_client, destination.strip(), interests, limit, api_key
        )
