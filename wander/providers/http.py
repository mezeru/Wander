from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import TravelConfigurationError, TravelProviderError


# HTTPX's INFO request log includes query parameters, which contain provider keys.
logging.getLogger("httpx").setLevel(logging.WARNING)


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    provider_name: str,
) -> Any:
    try:
        response = await client.get(url, params=params)
    except httpx.TimeoutException:
        raise TravelProviderError(
            f"{provider_name} request timed out.", transient=True
        ) from None
    except httpx.RequestError:
        raise TravelProviderError(
            f"Could not connect to {provider_name}.", transient=True
        ) from None

    if response.status_code in {401, 403}:
        raise TravelConfigurationError(
            f"{provider_name} rejected the API key; verify the key and its permissions."
        )
    if response.status_code == 429:
        raise TravelProviderError(
            f"{provider_name} rate limit reached; try again later.",
            transient=True,
        )
    if response.status_code >= 500:
        raise TravelProviderError(
            f"{provider_name} is temporarily unavailable (HTTP {response.status_code}).",
            transient=True,
        )
    if response.is_error:
        raise TravelProviderError(
            f"{provider_name} rejected the request (HTTP {response.status_code})."
        )

    try:
        return response.json()
    except (ValueError, UnicodeDecodeError):
        raise TravelProviderError(
            f"{provider_name} returned malformed JSON."
        ) from None
