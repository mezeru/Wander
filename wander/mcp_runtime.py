from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.client.stdio import StdioServerParameters


_TRAVEL_ENVIRONMENT = (
    "TRAVEL_DATA_PROVIDER",
    "TRAVEL_HTTP_TIMEOUT_SECONDS",
    "TRAVEL_API_MAX_RESULTS",
    "TRAVEL_SEARCH_RADIUS_METERS",
    "TRAVEL_FALLBACK_RADIUS_METERS",
    "TRAVEL_SEARCH_CANDIDATE_MULTIPLIER",
    "TRAVEL_OPENTRIPMAP_MIN_RATE",
    "TRAVEL_OPENTRIPMAP_MIN_SPACING_METERS",
    "OPENTRIPMAP_API_KEY",
    "GEOAPIFY_API_KEY",
)


def server_parameters(project_root: Path) -> StdioServerParameters:
    load_dotenv(project_root / ".env")
    environment = {
        name: os.environ[name]
        for name in _TRAVEL_ENVIRONMENT
        if name in os.environ
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "wander.mcp_server"],
        cwd=str(project_root),
        env=environment,
    )
