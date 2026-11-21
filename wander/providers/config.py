from __future__ import annotations

import math
import os
from dataclasses import dataclass


class TravelProviderError(RuntimeError):
    """A safe, user-displayable travel provider failure."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class TravelConfigurationError(TravelProviderError):
    """Required provider configuration is missing or invalid."""


class DestinationNotFoundError(TravelProviderError):
    """A provider could not resolve the requested destination safely."""


class AmbiguousDestinationError(TravelProviderError):
    """A provider returned multiple plausible destination matches."""


@dataclass(frozen=True)
class TravelSettings:
    provider: str
    timeout_seconds: float
    max_results: int
    search_radius_meters: int
    fallback_radius_meters: int
    candidate_multiplier: int
    opentripmap_min_rate: int
    opentripmap_min_spacing_meters: int

    def candidate_limit(self, target: int) -> int:
        return min(500, target * self.candidate_multiplier)


def travel_settings() -> TravelSettings:
    provider = os.getenv("TRAVEL_DATA_PROVIDER", "mock").strip().casefold()
    if provider not in {"mock", "live"}:
        raise TravelConfigurationError(
            "TRAVEL_DATA_PROVIDER must be either 'mock' or 'live'."
        )

    try:
        timeout = float(os.getenv("TRAVEL_HTTP_TIMEOUT_SECONDS", "15"))
        max_results = int(os.getenv("TRAVEL_API_MAX_RESULTS", "20"))
        radius = int(os.getenv("TRAVEL_SEARCH_RADIUS_METERS", "10000"))
        fallback_radius = int(os.getenv("TRAVEL_FALLBACK_RADIUS_METERS", "15000"))
        candidate_multiplier = int(os.getenv("TRAVEL_SEARCH_CANDIDATE_MULTIPLIER", "12"))
        min_rate = int(os.getenv("TRAVEL_OPENTRIPMAP_MIN_RATE", "2"))
        min_spacing = int(os.getenv("TRAVEL_OPENTRIPMAP_MIN_SPACING_METERS", "250"))
    except ValueError as exc:
        raise TravelConfigurationError(
            "Travel timeout, result limits, and search settings must be numeric."
        ) from exc

    if not math.isfinite(timeout) or min(
        timeout, max_results, radius, fallback_radius, candidate_multiplier,
    ) <= 0:
        raise TravelConfigurationError(
            "Travel timeout, result limit, search radii, and candidate multiplier must be positive."
        )
    if min_rate not in (1, 2, 3) or min_spacing < 0:
        raise TravelConfigurationError(
            "OpenTripMap minimum rate must be 1, 2, or 3 and minimum spacing must be nonnegative."
        )
    return TravelSettings(
        provider, timeout, max_results, radius, fallback_radius,
        candidate_multiplier, min_rate, min_spacing,
    )


def required_api_key(environment_name: str, provider_name: str) -> str:
    api_key = os.getenv(environment_name, "").strip()
    if not api_key:
        raise TravelConfigurationError(
            f"{environment_name} is required when live {provider_name} research is enabled."
        )
    return api_key
