from types import SimpleNamespace

from wander.graph.nodes.requirements_check import create_requirements_check_node
from wander.graph.nodes.routes import route_requirements


ACTIVITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "interests": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer"},
    },
    "required": ["destination"],
}

HOTEL_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "check_in": {"type": "string", "format": "date"},
        "check_out": {"type": "string", "format": "date"},
        "budget": {"type": "number"},
        "limit": {"type": "integer"},
    },
    "required": ["destination", "check_in", "check_out", "budget"],
}


def check(jobs, answers=None):
    context = SimpleNamespace(
        schemas={
            "search_attractions": ACTIVITIES_SCHEMA,
            "search_hotels": HOTEL_SCHEMA,
        }
    )
    node = create_requirements_check_node(context)
    state = {
        "research_jobs": jobs,
        "job_answers": answers or {},
    }
    return node(state)


def test_single_valid_activities_job():
    jobs = [{
        "job_id": "activities-0",
        "agent": "activities",
        "destination": "Tokyo",
        "tool_name": "search_attractions",
        "task": "Research attractions",
        "arguments": {"destination": "Tokyo", "limit": 5},
    }]
    result = check(jobs)
    assert result["missing_details"] == []
    assert result["validated_jobs"][0]["arguments"]["limit"] == 5


def test_two_hotel_jobs_are_validated_independently():
    jobs = [
        {"job_id": "accommodation-0", "destination": "Tokyo", "tool_name": "search_hotels",
         "arguments": {"destination": "Tokyo", "budget": 150}},
        {"job_id": "accommodation-1", "destination": "Kyoto", "tool_name": "search_hotels",
         "arguments": {"destination": "Kyoto", "budget": 150}},
    ]
    result = check(jobs)
    assert len(result["missing_details"]) == 4
    assert {item["job_id"] for item in result["missing_details"]} == {
        "accommodation-0", "accommodation-1",
    }


def test_clarifications_apply_to_correct_job_and_clear_errors():
    jobs = [
        {"job_id": "accommodation-0", "destination": "Tokyo", "tool_name": "search_hotels",
         "arguments": {"destination": "Tokyo", "budget": 150}},
        {"job_id": "accommodation-1", "destination": "Kyoto", "tool_name": "search_hotels",
         "arguments": {"destination": "Kyoto", "budget": 150}},
    ]
    result = check(jobs, {
        "accommodation-0": {"check_in": "2027-04-10", "check_out": "2027-04-12"},
        "accommodation-1": {"check_in": "2027-04-12", "check_out": "2027-04-14"},
    })
    assert result["missing_details"] == []
    by_id = {job["job_id"]: job for job in result["validated_jobs"]}
    assert by_id["accommodation-0"]["arguments"]["check_in"] == "2027-04-10"
    assert by_id["accommodation-1"]["arguments"]["check_in"] == "2027-04-12"


def test_invalid_and_reversed_dates_report_job_specific_errors():
    job = {"job_id": "accommodation-0", "destination": "Tokyo",
           "tool_name": "search_hotels",
           "arguments": {"destination": "Tokyo", "budget": 150,
                         "check_in": "04/12/2027", "check_out": "04/10/2027"}}
    result = check([job])
    fields = {(item["job_id"], item["field"]) for item in result["missing_details"]}
    assert ("accommodation-0", "check_in") not in fields
    assert ("accommodation-0", "check_out") in fields


def test_invalid_date_format_and_partial_corrections():
    job = {"job_id": "accommodation-0", "destination": "Tokyo",
           "tool_name": "search_hotels",
           "arguments": {"destination": "Tokyo", "budget": 150,
                         "check_in": "tomorrow", "check_out": "2027-04-12",
                         "limit": 3}}
    result = check([job])
    fields = {item["field"] for item in result["missing_details"]}
    assert "check_in" in fields
    validated = result["validated_jobs"][0]["arguments"]
    assert validated["budget"] == 150
    assert validated["check_out"] == "2027-04-12"
    assert validated["limit"] == 3


def test_unknown_tool_and_empty_jobs():
    unknown = check([{"job_id": "bad-0", "destination": "Tokyo",
                     "tool_name": "unsupported", "arguments": {"x": 1}}])
    assert unknown["missing_details"][0]["field"] == "_tool"
    empty = check([])
    assert empty["validated_jobs"] == []
    assert empty["missing_details"] == []
    assert route_requirements({
        "research_jobs": [],
        "missing_details": [],
    }) == "needs_input"


def test_no_dispatch_with_unresolved_fields():
    job = {"job_id": "activities-0", "destination": "Tokyo",
           "tool_name": "search_attractions", "arguments": {}}
    result = check([job])
    assert result["missing_details"]
    assert route_requirements({
        "research_jobs": [job],
        "missing_details": result["missing_details"],
        "clarification_rounds": 0,
    }) == "ask_user"
