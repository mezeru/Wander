# Wander

Wander is a learning-focused MVP for a multi-agent travel planner. The first milestone focuses on building a mock travel-data MCP server with realistic local datasets and reliable tool contracts.

## Included in this milestone

- A standalone MCP server exposing travel tools
- Local JSON-based mock destination data
- A sample MCP client for tool discovery and invocation
- A lightweight project layout for future agents and orchestration

## Quick start

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the MCP server
   ```bash
   python -m wander.mcp_server
   ```

4. Test the client against the server
   ```bash
   python -m wander.client
   ```

## Available tools

- `search_attractions(destination, interests=None, limit=5)`
- `search_hotels(destination, check_in, check_out, budget, limit=5)`
- `get_destination_info(destination)`
- `get_travel_times(origin, destination, transport_mode)`

## MCP resources

The server also exposes destination guide resources such as:

- `travel://destinations/Tokyo`
- `travel://destinations/Kyoto`

These allow MCP clients to load structured destination metadata without invoking a tool.

## Roadmap

This repo is intentionally minimal and designed to be extended toward:

- a LangGraph-powered Activities agent
- multi-agent trip coordination
- itinerary budget validation
- deeper MCP integrations and real travel APIs
