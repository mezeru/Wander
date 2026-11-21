import asyncio
from types import SimpleNamespace

import pytest

from wander.graph.nodes.routes import MAX_PLAN_ATTEMPTS, route_review
from wander.graph.nodes.validate_final import create_validate_final_node
from wander.graph.schemas import TripItinerary


def itinerary(days, activity="Shibuya Crossing", hotel="Shibuya Central Hotel"):
    return {
        "title": "Test trip",
        "days": [
            {
                "day_number": day_number,
                "destination": destination,
                "activities": [
                    {
                        "name": activity if destination == "Tokyo" else f"{destination} activity",
                        "evidence_job_id": f"activities-{day_number - 1}",
                    }
                ],
                "accommodation": hotel if destination == "Tokyo" else None,
                "accommodation_evidence_job_id": (
                    "accommodation-0" if destination == "Tokyo" else None
                ),
            }
            for day_number, destination in enumerate(days, start=1)
        ],
    }


def context(days=7):
    return {
        "destinations": ["Tokyo", "Kyoto", "Osaka"],
        "total_days": days,
        "day_allocation": {"Tokyo": 3, "Kyoto": 2, "Osaka": 2},
    }


def jobs():
    return [
        {"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"},
        {"job_id": "activities-1", "agent": "activities", "destination": "Tokyo"},
        {"job_id": "activities-2", "agent": "activities", "destination": "Tokyo"},
        {"job_id": "accommodation-0", "agent": "accommodation", "destination": "Tokyo"},
        {"job_id": "activities-3", "agent": "activities", "destination": "Kyoto"},
        {"job_id": "activities-4", "agent": "activities", "destination": "Kyoto"},
        {"job_id": "activities-5", "agent": "activities", "destination": "Osaka"},
        {"job_id": "activities-6", "agent": "activities", "destination": "Osaka"},
    ]


def structured_result(job_id, agent, destination, content):
    return {
        "job_id": job_id,
        "agent": agent,
        "destination": destination,
        "status": "completed",
        "result": content,
    }


def state_for(plan, *, jobs_value=None, results=None, trip_days=7, coverage=None):
    state = {
        "structured_plan": plan,
        "itinerary_context": context(trip_days),
        "validated_jobs": jobs_value if jobs_value is not None else [],
        "research_results": results or [],
        "trip_intent": {
            "destinations": ["Tokyo", "Kyoto", "Osaka"],
            "trip_days": 7,
            "nightly_hotel_budget": 150,
        },
    }
    if coverage is not None:
        state["research_coverage"] = coverage
    return state


def valid_empty_plan():
    sequence = ["Tokyo", "Tokyo", "Tokyo", "Kyoto", "Kyoto", "Osaka", "Osaka"]
    return {
        "title": "Valid allocation",
        "days": [
            {"day_number": index, "destination": destination, "activities": []}
            for index, destination in enumerate(sequence, start=1)
        ],
    }


def validate(state):
    fake_context = SimpleNamespace(
        validator_model=SimpleNamespace(
            ainvoke=lambda *_args, **_kwargs: pytest.fail("validator LLM was invoked")
        )
    )
    return asyncio.run(create_validate_final_node(fake_context)(state))


def test_valid_seven_day_structured_itinerary_is_verified():
    plan = {
        "title": "Tokyo Kyoto Osaka",
        "days": [
            {"day_number": 1, "destination": "Tokyo", "activities": [{"name": "Shibuya Crossing", "evidence_job_id": "activities-0"}]},
            {"day_number": 2, "destination": "Tokyo", "activities": [{"name": "Senso-ji Temple", "evidence_job_id": "activities-0"}]},
            {"day_number": 3, "destination": "Tokyo", "activities": [{"name": "Meiji Shrine", "evidence_job_id": "activities-0"}], "accommodation": "Shibuya Central Hotel", "accommodation_evidence_job_id": "accommodation-0"},
            {"day_number": 4, "destination": "Kyoto", "activities": [{"name": "Fushimi Inari Taisha", "evidence_job_id": "activities-1"}]},
            {"day_number": 5, "destination": "Kyoto", "activities": [{"name": "Kiyomizu-dera", "evidence_job_id": "activities-1"}]},
            {"day_number": 6, "destination": "Osaka", "activities": [{"name": "Osaka Castle", "evidence_job_id": "activities-2"}]},
            {"day_number": 7, "destination": "Osaka", "activities": [{"name": "Dotonbori", "evidence_job_id": "activities-2"}]},
        ],
    }
    jobs_value = [
        {"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"},
        {"job_id": "activities-1", "agent": "activities", "destination": "Kyoto"},
        {"job_id": "activities-2", "agent": "activities", "destination": "Osaka"},
        {"job_id": "accommodation-0", "agent": "accommodation", "destination": "Tokyo"},
    ]
    results = [
        structured_result("activities-0", "activities", "Tokyo", {"attractions": [{"name": "Shibuya Crossing"}, {"name": "Senso-ji Temple"}, {"name": "Meiji Shrine"}]}),
        structured_result("activities-1", "activities", "Kyoto", {"attractions": [{"name": "Fushimi Inari Taisha"}, {"name": "Kiyomizu-dera"}]}),
        structured_result("activities-2", "activities", "Osaka", {"attractions": [{"name": "Osaka Castle"}, {"name": "Dotonbori"}]}),
        structured_result("accommodation-0", "accommodation", "Tokyo", {"hotels": [{"name": "Shibuya Central Hotel", "nightly_rate": 145, "currency": "USD"}]}),
    ]
    state = state_for(plan, jobs_value=jobs_value, results=results)
    state["trip_intent"]["currency"] = "USD"
    result = validate(state)
    assert result["status"] == "verified"
    assert result["issues"] == []
    assert result["research_warnings"] == []


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ({"title": "Invalid", "days": []}, "6" if False else "exactly 7"),
        ({"title": "Missing", "days": [{"day_number": 1, "destination": "Tokyo", "activities": []}] * 7}, "destination"),
    ],
)
def test_incorrect_day_count_and_destination_allocation_are_invalid(plan, message):
    result = validate(state_for(plan))
    assert result["status"] == "invalid"
    assert any(message in issue.lower() for issue in result["issues"])


def test_missing_duplicate_and_skipped_day_numbers():
    plan = {"title": "Bad", "days": [
        {"day_number": 1, "destination": "Tokyo", "activities": []},
        {"day_number": 1, "destination": "Tokyo", "activities": []},
        {"day_number": 3, "destination": "Tokyo", "activities": []},
    ] + [{"day_number": n, "destination": "Kyoto", "activities": []} for n in (4, 5)] + [{"day_number": n, "destination": "Osaka", "activities": []} for n in (6, 7)]}
    result = validate(state_for(plan))
    assert result["status"] == "invalid"
    assert any("duplicate day" in issue.lower() for issue in result["issues"])


def test_missing_structured_plan_and_invalid_pydantic_object():
    missing = validate(state_for(None))
    assert missing["status"] == "invalid"
    assert "missing" in missing["issues"][0].lower()
    invalid = validate(state_for({"title": "Bad"}))
    assert invalid["status"] == "invalid"
    assert "invalid" in invalid["issues"][0].lower()


def test_unknown_job_missing_reference_and_wrong_destination_are_issues():
    plan = {
        "title": "Bad evidence",
        "days": [{"day_number": n, "destination": "Tokyo", "activities": [{"name": "Unknown", "evidence_job_id": "unknown"}]} for n in range(1, 8)],
    }
    result = validate(state_for(plan))
    assert result["status"] == "invalid"
    assert any("unknown evidence job" in issue.lower() for issue in result["issues"])

    plan["days"][0]["activities"][0]["evidence_job_id"] = ""
    result = validate(state_for(plan))
    assert any("no evidence_job_id" in issue.lower() for issue in result["issues"])


def test_failed_no_data_missing_and_duplicate_research_are_partial():
    plan = valid_empty_plan()
    state = state_for(
        plan,
        coverage={
            "failed": ["activities-0"],
            "no_data": ["activities-1"],
            "missing": ["activities-2"],
            "duplicate": ["activities-3"],
        },
    )
    result = validate(state)
    assert result["status"] == "partial"
    assert result["issues"] == []
    assert len(result["research_warnings"]) >= 4


def test_free_text_requires_verification_warning():
    plan = valid_empty_plan()
    plan["days"][0]["activities"] = [{"name": "Museum", "evidence_job_id": "activities-0"}]
    state = state_for(
        plan,
        jobs_value=[{"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"}],
        results=[structured_result("activities-0", "activities", "Tokyo", "Museum is popular.")],
    )
    result = validate(state)
    assert result["status"] == "partial"
    assert any("free-form" in warning for warning in result["research_warnings"])


def test_structured_entity_disproof_is_invalid():
    plan = {"title": "Bad", "days": [{"day_number": n, "destination": "Tokyo", "activities": [{"name": "Invented Place", "evidence_job_id": "activities-0"}]} for n in range(1, 8)]}
    state = state_for(plan, jobs_value=[
        {"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"},
    ], results=[structured_result("activities-0", "activities", "Tokyo", {"attractions": [{"name": "Shibuya Crossing"}]})])
    result = validate(state)
    assert result["status"] == "invalid"
    assert any("not present" in issue for issue in result["issues"])


def test_accommodation_must_reference_accommodation_job():
    plan = {"title": "Wrong category", "days": [{"day_number": n, "destination": "Tokyo", "activities": [], "accommodation": "Hotel", "accommodation_evidence_job_id": "activities-0"} for n in range(1, 8)]}
    state = state_for(plan, jobs_value=[
        {"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"},
    ], results=[structured_result("activities-0", "activities", "Tokyo", {"attractions": []})])
    result = validate(state)
    assert result["status"] == "invalid"
    assert any("not an 'accommodation' job" in issue for issue in result["issues"])


def test_budget_violation_and_currency_mismatch():
    plan = valid_empty_plan()
    plan["days"][0]["accommodation"] = "Ginza Budget Suites"
    plan["days"][0]["accommodation_evidence_job_id"] = "accommodation-0"
    jobs_value = [{"job_id": "accommodation-0", "agent": "accommodation", "destination": "Tokyo"}]
    record = {"hotels": [{"name": "Ginza Budget Suites", "nightly_rate": 185, "currency": "USD"}]}
    state = state_for(plan, jobs_value=jobs_value, results=[structured_result("accommodation-0", "accommodation", "Tokyo", record)])
    state["trip_intent"]["currency"] = "USD"
    state["trip_intent"]["nightly_hotel_budget"] = 150
    violated = validate(state)
    assert violated["status"] == "invalid"
    assert any("exceeding" in issue for issue in violated["issues"])

    state["trip_intent"]["currency"] = "EUR"
    incomparable = validate(state)
    assert incomparable["status"] == "partial" or incomparable["status"] == "verified"
    assert not any("exceeding" in issue for issue in incomparable["issues"])


def test_duplicate_result_and_mismatched_metadata():
    plan = valid_empty_plan()
    plan["days"][0]["activities"] = [{"name": "Place", "evidence_job_id": "activities-0"}]
    state = state_for(plan, jobs_value=[
        {"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"},
    ], results=[
        structured_result("activities-0", "activities", "Tokyo", {"attractions": [{"name": "Place"}]}),
        structured_result("activities-0", "activities", "Tokyo", {"attractions": [{"name": "Place"}]}),
    ])
    duplicate = validate(state)
    assert duplicate["status"] == "partial"
    assert any("duplicate" in warning.lower() for warning in duplicate["research_warnings"])

    mismatched_state = state_for(
        plan,
        jobs_value=[
            {"job_id": "activities-0", "agent": "activities", "destination": "Tokyo"},
        ],
        results=[
            structured_result("activities-0", "activities", "Kyoto", {"attractions": [{"name": "Place"}]}),
        ],
    )
    mismatched = validate(mismatched_state)
    assert mismatched["status"] == "invalid"
    assert any("mismatches" in issue for issue in mismatched["issues"])


def test_route_only_retries_correctable_issues_with_attempts_remaining():
    assert route_review({"issues": ["bad"], "plan_attempts": 0, "status": "invalid"}) == "review_again"
    assert route_review({"issues": ["bad"], "plan_attempts": MAX_PLAN_ATTEMPTS, "status": "invalid"}) == "__end__"
    assert route_review({"issues": [], "research_warnings": ["missing"], "status": "partial", "plan_attempts": 0}) == "__end__"
