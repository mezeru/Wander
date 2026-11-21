# Wander

Wander is a learning-focused MVP for a multi-agent travel planner. The first milestone focuses on building a mock travel-data MCP server with realistic local datasets and reliable tool contracts.

## Included in this milestone

- A standalone MCP server exposing travel tools
- Local JSON-based mock destination data
- A sample MCP client for tool discovery and invocation
- A lightweight project layout for future agents and orchestration

## Quick start

Run these commands from the project root (the directory containing `pyproject.toml`).

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the project and its dependencies
   ```bash
   python -m pip install -e .
   ```

3. Run the MCP server
   ```bash
   .venv\Scripts\python.exe -m wander.mcp_server
   ```

4. Test the client against the server
   ```bash
   .venv\Scripts\python.exe -m wander.client
   ```

## Workflow logging

Run the trip planner from the project root with `python -m wander.graph.trip_coordinator`.
A blank request selects the Kyoto demo. A trip plan asks for missing destinations
or total days before dispatching research. Attraction and lodging lookups return
research directly, and lodging lookups still ask for any tool-required dates
or nightly budget.

When every requested stop has a validated lodging stay, the itinerary calendar
starts at the first check-in. A transfer date belongs to the next stop, and extra
itinerary days go to the final stop. Only dates from check-in through the night
before check-out are lodging nights. The requested duration and hotel dates stay
independent, and incompatible dates or duration trigger clarification.

The trip planner emits INFO logs for node start/completion (including elapsed time),
route choices, worker outcomes, and validation issues. Use `WANDER_DEBUG=1` for
local development to log the exact request, extracted intent, validated tool
arguments, and returned and summarized place counts. Leave it off when logs
could contain personal travel details. A partial itinerary lists research
limitations separately from correctable planning issues.

## Graph code layout

- `wander/graph/schemas.py` contains the graph state and model schemas.
- `wander/graph/validation.py` contains tool-argument normalization and validation.
- `wander/graph/nodes/` contains one module per graph node, with shared dependencies
  in `context.py` and conditional routing functions in `routes.py`.
- `nodes/itinerary_context.py` groups usable research by destination and allocates
  initial city-days deterministically before the Final Planner.
- `wander/graph/trip_coordinator.py` assembles and runs the graph.

## Available tools

- `search_attractions(destination, interests=None, limit=5)`
- `search_hotels(destination, check_in, check_out, budget, limit=5)`
- `get_destination_info(destination)`
- `get_travel_times(origin, destination, transport_mode)`

## Travel research providers

The MCP tools keep their existing argument schemas and return lists of records.
Select the data source with `TRAVEL_DATA_PROVIDER`:

- `mock` uses the bundled local destination data and needs no API keys.
- `live` uses OpenTripMap for attractions and Geoapify for accommodations. A
  missing key or provider failure is reported; live mode never falls back to
  mock data. When unset, the provider defaults to `mock` to preserve offline
  development behavior.

Copy the example configuration, then add your own keys:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

Obtain API keys from [OpenTripMap](https://dev.opentripmap.org/) and
[Geoapify](https://myprojects.geoapify.com/), then fill in the key values in
`.env`. If `.env` already exists, preserve its contents and add the
missing variables. Set `TRAVEL_DATA_PROVIDER=live` and start the MCP server:

```powershell
python -m wander.mcp_server
```

Example tool inputs retain their existing signatures:

```text
search_attractions(destination="Tokyo")
search_hotels(destination="Kyoto", check_in="2026-04-10",
              check_out="2026-04-13", budget=200)
```

| Setting | Default | Purpose |
| --- | ---: | --- |
| `TRAVEL_API_MAX_RESULTS` | 20 | Maximum number of results returned per search. |
| `TRAVEL_SEARCH_RADIUS_METERS` | 10000 | Initial radius from the resolved city center. |
| `TRAVEL_FALLBACK_RADIUS_METERS` | 15000 | Radius for a wider search when results are sparse. A larger initial radius takes precedence. |
| `TRAVEL_SEARCH_CANDIDATE_MULTIPLIER` | 12 | Candidates requested per final result, capped at 500 per discovery request. |
| `TRAVEL_OPENTRIPMAP_MIN_RATE` | 2 | Initial OpenTripMap popularity threshold. Accepted values are `1`, `2`, and `3`. |
| `TRAVEL_OPENTRIPMAP_MIN_SPACING_METERS` | 250 | Preferred spacing between attractions. Use `0` to disable the preference. |

OpenTripMap keeps requested interests on every pass. It widens the radius and
then tries the provider's minimum rating of `1` when needed. When a full batch
falls well short of the search radius, it searches up to four additional areas
inside that radius. It ranks named places using provider rating and tourist
metadata, and prefers separated locations when the candidate pool allows.
Requests are paced within each server process. Entrance fees, visit durations,
and current opening hours may be unavailable. See the [OpenTripMap radius example](https://dev.opentripmap.org/examples).

Geoapify starts with hotels, hostels, and guesthouses. One fallback can widen
the circle and include other accommodation types. Dates and budget are returned
as user constraints. The Places API provides property locations, and room prices
and availability remain unverified. See the [Geoapify Places API documentation](https://apidocs.geoapify.com/docs/places/).

Both searches use circles around the provider's city center, so coverage varies
with city size and provider data. Discovery counts and fallback decisions appear
in server logs on stderr. Provider usage limits and attribution requirements apply.

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
