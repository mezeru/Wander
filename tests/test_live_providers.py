import asyncio
import inspect
import logging

import httpx
import pytest

from wander.graph.mcp_schemas import Attraction, Hotel
from wander.providers import opentripmap
from wander.providers.config import (
    AmbiguousDestinationError,
    DestinationNotFoundError,
    TravelConfigurationError,
    TravelProviderError,
)
from wander.providers.geoapify import search_hotels
from wander.providers.opentripmap import _popularity, search_attractions


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run_search(handler, search, *args, **kwargs):
    async def run():
        async with _client(handler) as client:
            return await search(*args, client=client, **kwargs)

    return asyncio.run(run())


def _opentripmap_handler(
    *,
    names=("Senso-ji",),
    records=None,
    detail=None,
):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/geoname"):
            return httpx.Response(
                200,
                json={"name": "Tokyo", "lat": 35.68, "lon": 139.69},
            )
        if request.url.path.endswith("/radius"):
            if records is not None:
                payload = records(request.url.params) if callable(records) else records
                return httpx.Response(200, json=payload)
            return httpx.Response(
                200,
                json=[
                    {"xid": f"xid-{index}", "name": name, "rate": index}
                    for index, name in enumerate(names, start=1)
                ],
            )
        xid = request.url.path.rsplit("/", 1)[-1]
        if detail is not None:
            return httpx.Response(200, json=detail(xid) if callable(detail) else detail)
        return httpx.Response(
            200,
            json={
                "xid": xid,
                "name": f"Place {xid}",
                "kinds": "historic, cultural",
                "wikipedia_extracts": {"text": "Useful description"},
                "point": {"lat": 35.7, "lon": 139.7},
                "url": "https://example.test/place",
            },
        )

    return handler, calls


def _geoapify_handler(*, features=None, geocode=None):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/geocode/search"):
            if geocode is not None:
                return httpx.Response(200, json={"features": geocode})
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "properties": {
                                "city": "Kyoto",
                                "country": "Japan",
                            },
                            "geometry": {"coordinates": [135.7681, 35.0116]},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"features": features(request.url.params) if callable(features) else features or []},
        )

    return handler, calls


def _place(place_id, name="Kyoto Hotel", **overrides):
    properties = {
        "place_id": place_id,
        "name": name,
        "categories": ["accommodation.hotel"],
        "formatted": "1 Kyoto Street, Kyoto",
        "website": "https://example.test/hotel",
    }
    properties.update(overrides)
    return {
        "properties": properties,
        "geometry": {"coordinates": [135.77, 35.01]},
    }


def _configure(monkeypatch, provider="live"):
    monkeypatch.setenv("TRAVEL_DATA_PROVIDER", provider)
    monkeypatch.setenv("TRAVEL_HTTP_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("TRAVEL_API_MAX_RESULTS", "20")
    monkeypatch.setenv("TRAVEL_SEARCH_RADIUS_METERS", "12000")
    monkeypatch.setenv("TRAVEL_FALLBACK_RADIUS_METERS", "15000")
    monkeypatch.setenv("TRAVEL_SEARCH_CANDIDATE_MULTIPLIER", "12")
    monkeypatch.setenv("TRAVEL_OPENTRIPMAP_MIN_RATE", "2")
    monkeypatch.setenv("TRAVEL_OPENTRIPMAP_MIN_SPACING_METERS", "250")
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "unit-test-opentripmap-key")
    monkeypatch.setenv("GEOAPIFY_API_KEY", "unit-test-geoapify-key")


def test_opentripmap_resolves_searches_details_and_normalizes(monkeypatch):
    _configure(monkeypatch)
    handler, calls = _opentripmap_handler()
    with asyncio.Runner() as runner:
        async def run():
            async with _client(handler) as client:
                return await search_attractions(
                    "Tokyo", ["culture"], 1, client=client
                )
        results = runner.run(run())

    assert len(results) == 1
    result = results[0]
    assert result["id"] == "xid-1"
    assert result["name"] == "Place xid-1"
    assert result["destination"] == "Tokyo"
    assert result["provider"] == "opentripmap"
    assert result["coordinates"] == {"latitude": 35.7, "longitude": 139.7}
    assert result["description"] == "Useful description"
    assert result["estimated_duration_hours"] is None
    assert result["price"] is None
    assert result["verified_live"] is False
    radius = next(request for request in calls if request.url.path.endswith("/radius"))
    assert radius.url.params["radius"] == "12000"
    assert radius.url.params["kinds"] == "cultural"
    assert radius.url.params["limit"] == "12"
    assert radius.url.params["rate"] == "2"
    assert len([r for r in calls if "/xid/" in r.url.path]) == 1


def test_opentripmap_missing_description_and_optional_values(monkeypatch):
    _configure(monkeypatch)
    handler, _ = _opentripmap_handler(detail={
        "xid": "xid-1",
        "name": "Place",
        "kinds": "interesting_places",
    })
    with asyncio.Runner() as runner:
        async def run():
            async with _client(handler) as client:
                return await search_attractions("Tokyo", client=client)
        result = runner.run(run())[0]
    assert result["description"] is None
    assert result["price"] is None
    assert result["estimated_duration_hours"] is None
    assert Attraction.model_validate(result).id == "xid-1"


def test_opentripmap_empty_results_and_maximum_limit(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("TRAVEL_API_MAX_RESULTS", "1")
    handler, calls = _opentripmap_handler(names=())
    with asyncio.Runner() as runner:
        async def run():
            async with _client(handler) as client:
                return await search_attractions("Tokyo", limit=10, client=client)
        result = runner.run(run())
    assert result == []
    searches = [request.url.params for request in calls if request.url.path.endswith("/radius")]
    assert [(p["radius"], p["rate"], p["limit"]) for p in searches] == [
        ("12000", "2", "12"), ("15000", "2", "12"), ("15000", "1", "12"),
    ]
    assert len(calls) == 4


@pytest.mark.parametrize(
    ("value", "normalized"),
    [("1", 1), (2, 2), ("3h", 3), (None, 0), ("bad", 0), ("NaN", 0), (True, 0)],
)
def test_opentripmap_normalizes_provider_ratings(value, normalized):
    assert _popularity(value) == normalized


def test_opentripmap_ranks_overfetched_records_and_caps_final_output(monkeypatch, caplog):
    _configure(monkeypatch)
    monkeypatch.setenv("TRAVEL_API_MAX_RESULTS", "3")
    monkeypatch.setenv("TRAVEL_SEARCH_CANDIDATE_MULTIPLIER", "4")
    monkeypatch.setenv("TRAVEL_OPENTRIPMAP_MIN_SPACING_METERS", "0")
    handler, calls = _opentripmap_handler(records=[
        {"xid": "missing", "name": "Missing rate"},
        {"xid": "string", "name": "String rate", "rate": "1"},
        {"xid": "malformed", "name": "Malformed rate", "rate": "bad"},
        {"xid": "unnamed", "name": " ", "rate": 3},
        {"name": "Missing ID", "rate": 3},
        {"xid": "numeric", "name": "Numeric rate", "rate": 2},
        {"xid": "heritage", "name": "Heritage rate", "rate": "3h"},
        {"xid": "heritage", "name": "Duplicate", "rate": "3h"},
    ])
    with caplog.at_level(logging.INFO):
        results = _run_search(handler, search_attractions, "Tokyo", limit=9)

    assert [item["id"] for item in results] == ["heritage", "numeric", "string"]
    assert results[-1]["provider_metadata"]["rate"] == "1"
    assert calls[1].url.params["limit"] == "12"
    assert "kinds" not in calls[1].url.params
    assert "raw=8 candidate_pool=5 rejected=2 duplicates=1" in caplog.text
    assert "fallback=False" in caplog.text
    assert "unit-test-opentripmap-key" not in caplog.text


def test_opentripmap_prefers_spaced_locations_and_description(monkeypatch):
    _configure(monkeypatch)
    records = [
        {"xid": "first", "name": "First", "rate": "3", "dist": 1,
         "point": {"lat": 35.68, "lon": 139.69}},
        {"xid": "duplicate-point", "name": "Co-located", "rate": "3", "dist": 2,
         "point": {"lat": 35.68, "lon": 139.69}},
        {"xid": "nearby", "name": "Nearby", "rate": "3", "dist": 5,
         "point": {"lat": 35.6805, "lon": 139.69}},
        {"xid": "further", "name": "Further", "rate": "2", "dist": 2000,
         "point": {"lat": 35.70, "lon": 139.69}},
        {"xid": "described", "name": "Described", "rate": "2", "dist": 4000,
         "point": {"lat": 35.72, "lon": 139.69}},
    ]

    def detail(xid):
        if xid == "described":
            return {"name": "  ", "info": {"descr": "An identifiable place"},
                    "point": {"lat": 120, "lon": 139.69}}
        return {}

    handler, calls = _opentripmap_handler(records=records, detail=detail)
    results = _run_search(handler, search_attractions, "Tokyo", limit=2)

    assert [item["id"] for item in results] == ["first", "described"]
    assert results[1]["name"] == "Described"
    assert results[1]["coordinates"] == {"latitude": 35.72, "longitude": 139.69}
    assert [r.url.path.rsplit("/", 1)[-1] for r in calls if "/xid/" in r.url.path] == [
        "first", "further", "described",
    ]


def test_opentripmap_fallback_merges_short_results_and_preserves_interests(monkeypatch):
    _configure(monkeypatch)
    strict = {"xid": "strict", "name": "Strict", "rate": "3"}

    def records(params):
        if params["rate"] == "1":
            return [strict, {"xid": "relaxed", "name": "Relaxed", "rate": "1"}]
        return [strict]

    handler, calls = _opentripmap_handler(records=records, detail={})
    results = _run_search(handler, search_attractions, "Tokyo", ["history", "museum"], limit=2)
    searches = [r.url.params for r in calls if r.url.path.endswith("/radius")]
    assert [(p["rate"], p["radius"], p["kinds"]) for p in searches] == [
        ("2", "12000", "historic,museums"),
        ("2", "15000", "historic,museums"),
        ("1", "15000", "historic,museums"),
    ]
    assert [item["id"] for item in results] == ["strict", "relaxed"]
    assert len([r for r in calls if "/xid/" in r.url.path]) == 2


def test_opentripmap_bounds_detail_requests_after_normalization_failures(monkeypatch):
    _configure(monkeypatch)

    def records(params):
        return [
            {"xid": f"{params['rate']}-{params['radius']}-{i}", "name": f"Place {i}"}
            for i in range(12)
        ]

    handler, calls = _opentripmap_handler(records=records, detail=[])
    assert _run_search(handler, search_attractions, "Tokyo", limit=2) == []
    details = [r.url.path for r in calls if "/xid/" in r.url.path]
    assert len(details) == len(set(details)) == 8
    assert len([r for r in calls if r.url.path.endswith("/radius")]) == 3


def test_opentripmap_samples_full_nearest_first_pool_only_once(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("TRAVEL_SEARCH_CANDIDATE_MULTIPLIER", "2")

    def records(params):
        lat, lon = float(params["lat"]), float(params["lon"])
        is_center = (lat, lon) == (35.68, 139.69)
        return [
            {"xid": f"{lat}-{lon}-{i}", "name": f"Place {i}", "dist": i + 1,
             "rate": "2" if is_center else "3", "point": {"lat": lat, "lon": lon}}
            for i in range(4 if is_center else 1)
        ]

    handler, calls = _opentripmap_handler(records=records, detail={})
    results = _run_search(handler, search_attractions, "Tokyo", ["culture"], limit=2)

    searches = [r.url.params for r in calls if r.url.path.endswith("/radius")]
    assert [p["radius"] for p in searches] == ["12000", "6000", "6000", "6000", "6000"]
    assert all(p["kinds"] == "cultural" and p["rate"] == "2" for p in searches)
    assert len({(p["lat"], p["lon"]) for p in searches}) == 5
    assert len(results) == 2
    assert all(item["provider_metadata"]["distance_from_center_meters"] == pytest.approx(3000)
               for item in results)


def test_opentripmap_spaces_live_request_slots_across_clients(monkeypatch):
    delays = []

    async def record_delay(delay):
        delays.append(delay)

    monkeypatch.setattr(opentripmap, "_next_request_at", 0)
    monkeypatch.setattr(opentripmap, "monotonic", lambda: 10.0)
    monkeypatch.setattr(opentripmap, "sleep", record_delay)

    async def run():
        async with (
            _client(lambda _request: httpx.Response(200)) as first,
            _client(lambda _request: httpx.Response(200)) as second,
        ):
            first.event_hooks["request"].append(opentripmap._pace_request)
            second.event_hooks["request"].append(opentripmap._pace_request)
            url = "https://example.test/radius"
            await asyncio.gather(first.get(url), second.get(url), first.get(url))

    asyncio.run(run())
    assert delays == [0.25, 0.5]


def test_opentripmap_rejects_unknown_interests_before_requests(monkeypatch):
    _configure(monkeypatch)
    handler, calls = _opentripmap_handler()
    with pytest.raises(TravelConfigurationError, match="Unsupported attraction interests"):
        _run_search(handler, search_attractions, "Tokyo", ["culture", "unknown"])
    assert not calls


def test_opentripmap_rejects_wrong_or_ambiguous_city(monkeypatch):
    _configure(monkeypatch)

    def wrong_city(request):
        return httpx.Response(200, json={"name": "Tokyo, Japan", "lat": 35, "lon": 139})

    with asyncio.Runner() as runner:
        async def run_wrong():
            async with _client(wrong_city) as client:
                await search_attractions("Tokyo", client=client)
        with pytest.raises(DestinationNotFoundError):
            runner.run(run_wrong())

    def ambiguous(request):
        return httpx.Response(200, json=[
            {"name": "Tokyo", "lat": 35, "lon": 139},
            {"name": "Tokyo", "lat": 36, "lon": 140},
        ])

    with asyncio.Runner() as runner:
        async def run_ambiguous():
            async with _client(ambiguous) as client:
                await search_attractions("Tokyo", client=client)
        with pytest.raises(AmbiguousDestinationError):
            runner.run(run_ambiguous())


def test_geoapify_geocodes_and_normalizes_properties_without_prices(monkeypatch):
    _configure(monkeypatch)
    handler, calls = _geoapify_handler(features=[_place("place-1")])
    with asyncio.Runner() as runner:
        async def run():
            async with _client(handler) as client:
                return await search_hotels(
                    "Kyoto", "2026-04-10", "2026-04-13", 200, limit=1, client=client
                )
        results = runner.run(run())

    assert len(results) == 1
    result = results[0]
    assert result["id"] == "place-1"
    assert result["destination"] == "Kyoto"
    assert result["provider"] == "geoapify"
    assert result["nightly_price"] is None
    assert result["pricing_available"] is False
    assert result["availability_verified"] is False
    assert result["date_availability_checked"] is False
    assert result["budget_constraint"] == 200
    assert result["check_in"] == "2026-04-10"
    assert result["check_out"] == "2026-04-13"
    assert result["coordinates"] == {"latitude": 35.01, "longitude": 135.77}
    assert Hotel.model_validate(result).nightly_price is None

    places = next(request for request in calls if request.url.path.endswith("/places"))
    assert places.url.params["categories"] == (
        "accommodation.hotel,accommodation.hostel,accommodation.guest_house"
    )
    assert places.url.params["filter"] == "circle:135.7681,35.0116,12000"
    assert places.url.params["limit"] == "12"
    assert places.url.params["apiKey"] == "unit-test-geoapify-key"


def test_geoapify_skips_unnamed_records_and_deduplicates_place_ids(monkeypatch):
    _configure(monkeypatch)
    handler, _ = _geoapify_handler(features=[
        _place("same-id"),
        _place("same-id", "Duplicate hotel"),
        _place("no-name", ""),
        _place("", "No stable ID"),
    ])
    with asyncio.Runner() as runner:
        async def run():
            async with _client(handler) as client:
                return await search_hotels(
                    "Kyoto", "2026-04-10", "2026-04-11", 100, client=client
                )
        results = runner.run(run())
    assert len(results) == 1
    assert results[0]["name"] == "Kyoto Hotel"


def test_geoapify_distinguishes_empty_features_from_normalization_rejections(monkeypatch, caplog):
    _configure(monkeypatch)
    invalid_geometry = _place("invalid-geometry")
    invalid_geometry["geometry"]["coordinates"] = [135.77, True]
    handler, calls = _geoapify_handler(features=lambda params: (
        [] if params["categories"] != "accommodation" else [
            _place("no-name", ""), _place(None), invalid_geometry, None,
        ]
    ))
    with caplog.at_level(logging.INFO):
        results = _run_search(handler, search_hotels, "Kyoto", "a", "b", 50)

    assert results == []
    assert len(calls) == 3
    assert "geocoding_candidates=1" in caplog.text
    assert "selected lat=35.0116 lon=135.7681" in caplog.text
    assert "fallback=False raw=0 missing_name=0" in caplog.text
    assert "fallback=True raw=4 missing_name=1 missing_place_id=1 invalid_geometry=1 malformed=1" in caplog.text
    assert "raw=4 rejected=4 normalized=0 duplicates=0 valid=0 fallback=True returned=0" in caplog.text
    assert "unit-test-geoapify-key" not in caplog.text


def test_geoapify_broadens_categories_and_radius_with_bounded_results(monkeypatch, caplog):
    _configure(monkeypatch)
    stays = _place("other-stay", "Kyoto apartment", categories=["accommodation.apartment"],
                   lon=135.76, lat=35.04)
    del stays["geometry"]
    geocode = [{"properties": {"city": "Kyoto", "lon": 135.77, "lat": 35.01}}]
    first = _place("first")
    handler, calls = _geoapify_handler(geocode=geocode, features=lambda params: (
        [first] if params["categories"] != "accommodation" else [
            first, _place("business", "Shop", categories=["commercial"]), stays,
            _place("behind-limit", "Another accommodation"),
        ]
    ))
    with caplog.at_level(logging.INFO):
        results = _run_search(handler, search_hotels, "Kyoto", "a", "b", 50, limit=2)

    assert [item["id"] for item in results] == ["first", "other-stay"]
    assert results[1]["coordinates"] == {"latitude": 35.04, "longitude": 135.76}
    assert len(calls) == 3
    assert [r.url.params["filter"] for r in calls[1:]] == [
        "circle:135.77,35.01,12000", "circle:135.77,35.01,15000",
    ]
    assert all(r.url.params["limit"] == "24" for r in calls[1:])
    assert "non_accommodation=1 normalized=3 duplicates=1" in caplog.text
    assert "valid=3 fallback=True returned=2" in caplog.text


def test_geoapify_widens_an_explicit_lodging_filter_without_changing_it(monkeypatch):
    _configure(monkeypatch)
    handler, calls = _geoapify_handler()
    assert _run_search(handler, search_hotels, "Kyoto", "a", "b", 50, interests=["hostel"]) == []
    assert [r.url.params["categories"] for r in calls[1:]] == [
        "accommodation.hostel", "accommodation.hostel",
    ]


def test_geoapify_empty_and_ambiguous_destinations(monkeypatch):
    _configure(monkeypatch)

    def empty(request):
        return httpx.Response(200, json={"features": []})

    with asyncio.Runner() as runner:
        async def run_empty():
            async with _client(empty) as client:
                await search_hotels("Unknown", "a", "b", 50, client=client)
        with pytest.raises(DestinationNotFoundError):
            runner.run(run_empty())

    def ambiguous(request):
        return httpx.Response(200, json={
            "features": [
                {"properties": {"city": "Kyoto", "country": "Japan"},
                 "geometry": {"coordinates": [135.7, 35.0]}},
                {"properties": {"city": "Kyoto", "country": "Japan"},
                 "geometry": {"coordinates": [136.0, 35.2]}},
            ]
        })

    with asyncio.Runner() as runner:
        async def run_ambiguous():
            async with _client(ambiguous) as client:
                await search_hotels("Kyoto", "a", "b", 50, client=client)
        with pytest.raises(AmbiguousDestinationError):
            runner.run(run_ambiguous())


@pytest.mark.parametrize(
    ("search", "key_name", "args"),
    [
        (search_attractions, "OPENTRIPMAP_API_KEY", ("Tokyo",)),
        (search_hotels, "GEOAPIFY_API_KEY", ("Kyoto", "2026-04-10", "2026-04-12", 200)),
    ],
)
def test_missing_provider_keys_are_actionable(monkeypatch, search, key_name, args):
    _configure(monkeypatch)
    monkeypatch.delenv(key_name, raising=False)
    with pytest.raises(TravelConfigurationError, match=key_name):
        asyncio.run(search(*args))


@pytest.mark.parametrize(
    ("search", "args", "key_name"),
    [
        (search_attractions, ("Tokyo",), "OPENTRIPMAP_API_KEY"),
        (search_hotels, ("Kyoto", "a", "b", 50), "GEOAPIFY_API_KEY"),
    ],
)
@pytest.mark.parametrize(
    ("status_code", "transient"),
    [(401, False), (429, True), (503, True)],
)
def test_provider_auth_rate_limit_and_server_errors_are_safe(
    monkeypatch, search, args, key_name, status_code, transient, caplog
):
    _configure(monkeypatch)
    secret = "unit-test-secret-never-log"
    monkeypatch.setenv(key_name, secret)

    def failure(request):
        return httpx.Response(status_code, text=f"key={secret}")

    with caplog.at_level(logging.INFO):
        with asyncio.Runner() as runner:
            async def run():
                async with _client(failure) as client:
                    return await search(*args, client=client)
            with pytest.raises(TravelProviderError) as error:
                runner.run(run())

    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert error.value.transient is transient


@pytest.mark.parametrize(
    ("search", "args", "key_name"),
    [
        (search_attractions, ("Tokyo",), "OPENTRIPMAP_API_KEY"),
        (search_hotels, ("Kyoto", "a", "b", 50), "GEOAPIFY_API_KEY"),
    ],
)
def test_provider_timeout_is_transient_and_does_not_leak_key(
    monkeypatch, search, args, key_name
):
    _configure(monkeypatch)
    secret = "timeout-key-secret"
    monkeypatch.setenv(key_name, secret)

    def timeout(request):
        raise httpx.ReadTimeout(f"timed out for {request.url}")

    with asyncio.Runner() as runner:
        async def run():
            async with _client(timeout) as client:
                return await search(*args, client=client)
        with pytest.raises(TravelProviderError) as error:
            runner.run(run())
    assert error.value.transient is True
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    ("search", "args", "key_name"),
    [
        (search_attractions, ("Tokyo",), "OPENTRIPMAP_API_KEY"),
        (search_hotels, ("Kyoto", "a", "b", 50), "GEOAPIFY_API_KEY"),
    ],
)
def test_provider_malformed_json_is_reported(monkeypatch, search, args, key_name):
    _configure(monkeypatch)
    monkeypatch.setenv(key_name, "dummy")

    def malformed(request):
        return httpx.Response(200, text="not-json", headers={"content-type": "application/json"})

    with asyncio.Runner() as runner:
        async def run():
            async with _client(malformed) as client:
                return await search(*args, client=client)
        with pytest.raises(TravelProviderError, match="malformed JSON"):
            runner.run(run())


def test_mcp_tool_signatures_remain_compatible(monkeypatch):
    _configure(monkeypatch, "mock")
    from wander import mcp_server

    attraction_parameters = inspect.signature(mcp_server.search_attractions).parameters
    hotel_parameters = inspect.signature(mcp_server.search_hotels_tool).parameters
    assert list(attraction_parameters) == ["destination", "interests", "limit"]
    assert list(hotel_parameters) == [
        "destination", "check_in", "check_out", "budget", "limit"
    ]


def test_mock_selection_uses_local_data_without_live_fallback(monkeypatch):
    _configure(monkeypatch, "mock")
    from wander import mcp_server

    monkeypatch.setattr(
        mcp_server,
        "get_attractions",
        lambda **_kwargs: [{"name": "Mock place"}],
    )
    monkeypatch.setattr(
        mcp_server,
        "search_hotels",
        lambda *_args: [{"name": "Mock hotel", "nightly_rate": 100}],
    )
    monkeypatch.setattr(
        mcp_server,
        "search_attractions_live",
        lambda *_args: pytest.fail("live provider called in mock mode"),
    )
    monkeypatch.setattr(
        mcp_server,
        "search_hotels_live",
        lambda *_args: pytest.fail("live provider called in mock mode"),
    )
    attractions = asyncio.run(mcp_server.search_attractions("Tokyo"))
    hotels = asyncio.run(
        mcp_server.search_hotels_tool("Kyoto", "a", "b", 200)
    )
    assert attractions[0]["provider"] == "mock"
    assert hotels[0]["provider"] == "mock"
    assert hotels[0]["pricing_available"] is True


def test_live_failure_does_not_fall_back_to_mock(monkeypatch):
    _configure(monkeypatch, "live")
    from wander import mcp_server

    def failed(*_args):
        raise TravelProviderError("OpenTripMap temporarily unavailable.")

    monkeypatch.setattr(mcp_server, "search_attractions_live", failed)
    monkeypatch.setattr(
        mcp_server,
        "get_attractions",
        lambda **_kwargs: pytest.fail("mock fallback was called"),
    )
    with pytest.raises(TravelProviderError, match="temporarily unavailable"):
        asyncio.run(mcp_server.search_attractions("Tokyo"))
