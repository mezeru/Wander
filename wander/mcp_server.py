from __future__ import annotations

import json
from typing import Any

from wander.travel_data import (
    get_destination_info,
    get_travel_times,
    search_hotels,
    get_attractions
)

from mcp.server.mcpserver import MCPServer
mcp = MCPServer("destination")


@mcp.tool()
def search_attractions(destination: str,
  interests: list[str] | None,
  limit: int) -> list[dict[str, Any]]:

    attractions = get_attractions(destination=destination, interests=interests, limit=limit)
    return attractions


@mcp.resource('travel://guides/{destination}')
def destination_guide(destination: str) -> dict[str, Any]:
    info = get_destination_info(destination=destination)
    return json.dumps(info, indent=2)

@mcp.tool()
def search_hotels(destination: str, start_date, end_date):
    hotels = {
        "Hilton":{
            "rating":4.5,
            "price": 200,
            "location": destination,
            "start_date": start_date,
            "end_date": end_date
        }
    }

    return hotels

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
    mcp.run(transport="stdio")