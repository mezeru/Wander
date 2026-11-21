from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

from mcp.server import MCPServer

from wander.providers.config import travel_settings
from wander.providers.geoapify import search_hotels as search_hotels_live
from wander.providers.opentripmap import search_attractions as search_attractions_live
from wander.travel_data import get_attractions, get_destination_info, search_hotels


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger(__name__)
mcp = MCPServer(name="destination")


@mcp.tool()
async def search_attractions(
    destination: str,
    interests: list[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    settings = travel_settings()
    if settings.provider == "live":
        return await search_attractions_live(destination, interests, limit)

    attractions = get_attractions(
        destination=destination,
        interests=interests,
        limit=limit,
    )
    return [
        {
            **item,
            "destination": destination,
            "provider": "mock",
            "source": "mock",
            "data_provider": "mock",
            "verified_live": False,
        }
        for item in attractions
    ]


@mcp.resource('travel://guides/{destination}')
def destination_guide(destination: str) -> dict[str, Any]:
    info = get_destination_info(destination=destination)
    return json.dumps(info, indent=2)

@mcp.tool(name="search_hotels")
async def search_hotels_tool(
    destination: str,
    check_in: str,
    check_out: str,
    budget: float,
    limit: int = 5
) -> list[dict[str, Any]]:
    settings = travel_settings()
    if settings.provider == "live":
        return await search_hotels_live(
            destination,
            check_in,
            check_out,
            budget,
            limit,
        )

    hotels = search_hotels(destination, check_in, check_out, budget, limit)
    return [
        {
            **item,
            "destination": destination,
            "provider": "mock",
            "source": "mock",
            "data_provider": "mock",
            "pricing_available": True,
            "availability_verified": False,
            "date_availability_checked": False,
        }
        for item in hotels
    ]

@mcp.prompt()
def trip_planning_prompt(
    destinations: str,
    days: int = 5
) -> str:

    cities = [
        city.strip()
        for city in destinations.split(",")
        if city.strip()
    ]

    destinations_str = " → ".join(cities)

    return f"""
    You are an expert travel planner.

    Create a detailed {days}-day itinerary
    for the following destinations: {destinations_str}.

    Include:
    - Day-by-day activities
    - Transportation between destinations
    - Food recommendations
    - Practical travel tips

    Ensure realistic pacing and minimize
    unnecessary travel.
    """
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="stdio")