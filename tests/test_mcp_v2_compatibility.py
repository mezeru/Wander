import asyncio
from pathlib import Path
import sys
import tomllib

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import MCPServer

from wander.mcp_runtime import server_parameters


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_server_import_uses_mcp_v2_server_class():
    from wander import mcp_server

    assert isinstance(mcp_server.mcp, MCPServer)
    assert mcp_server.MCPServer is MCPServer


def test_client_server_parameters_use_same_interpreter_and_allowlisted_env(
    monkeypatch,
):
    monkeypatch.setenv("TRAVEL_DATA_PROVIDER", "mock")
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "local-test-value")
    settings = {
        "TRAVEL_FALLBACK_RADIUS_METERS": "14000",
        "TRAVEL_SEARCH_CANDIDATE_MULTIPLIER": "8",
        "TRAVEL_OPENTRIPMAP_MIN_RATE": "3",
        "TRAVEL_OPENTRIPMAP_MIN_SPACING_METERS": "400",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-forwarded")

    parameters = server_parameters(PROJECT_ROOT)

    assert parameters.command == sys.executable
    assert parameters.cwd == str(PROJECT_ROOT)
    assert parameters.env["TRAVEL_DATA_PROVIDER"] == "mock"
    assert parameters.env["OPENTRIPMAP_API_KEY"] == "local-test-value"
    assert {name: parameters.env[name] for name in settings} == settings
    assert "UNRELATED_SECRET" not in parameters.env


def test_stdio_tools_resources_and_prompts_work_without_api_keys(monkeypatch):
    monkeypatch.setenv("TRAVEL_DATA_PROVIDER", "mock")
    monkeypatch.delenv("OPENTRIPMAP_API_KEY", raising=False)
    monkeypatch.delenv("GEOAPIFY_API_KEY", raising=False)
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "wander.mcp_server"],
        cwd=str(PROJECT_ROOT),
        env={"TRAVEL_DATA_PROVIDER": "mock", "PYTHON_DOTENV_DISABLED": "1"},
    )

    async def exercise_server():
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                prompts = await session.list_prompts()
                templates = await session.list_resource_templates()
                attraction_result = await session.call_tool(
                    "search_attractions",
                    {"destination": "Tokyo", "interests": ["culture"], "limit": 1},
                )
                hotel_result = await session.call_tool(
                    "search_hotels",
                    {
                        "destination": "Kyoto",
                        "check_in": "2026-04-10",
                        "check_out": "2026-04-12",
                        "budget": 200,
                        "limit": 1,
                    },
                )
                return tools, prompts, templates, attraction_result, hotel_result

    tools, prompts, templates, attraction_result, hotel_result = asyncio.run(
        exercise_server()
    )
    schemas = {tool.name: tool.input_schema for tool in tools.tools}

    assert set(schemas) == {"search_attractions", "search_hotels"}
    assert schemas["search_attractions"]["required"] == ["destination"]
    interests = schemas["search_attractions"]["properties"]["interests"]
    assert {
        "type": "array",
        "items": {"type": "string"},
    } in interests["anyOf"]
    assert schemas["search_attractions"]["properties"]["limit"]["type"] == "integer"
    assert set(schemas["search_hotels"]["required"]) == {
        "destination", "check_in", "check_out", "budget",
    }
    assert schemas["search_hotels"]["properties"]["limit"]["type"] == "integer"
    assert "trip_planning_prompt" in {prompt.name for prompt in prompts.prompts}
    assert any("guides" in item.uri_template for item in templates.resource_templates)
    assert attraction_result.is_error is False
    assert attraction_result.structured_content["result"][0]["provider"] == "mock"
    assert hotel_result.is_error is False
    assert hotel_result.structured_content["result"][0]["provider"] == "mock"


def test_trip_graph_imports_with_mcp_v2_error_type():
    from mcp import MCPError
    from wander.graph.nodes.research_worker import _is_retryable

    assert not _is_retryable(MCPError(-32602, "invalid parameters"))


def test_mcp_v1_imports_and_dependency_constraints_are_absent():
    python_files = (PROJECT_ROOT / "wander").rglob("*.py")
    application_source = "\n".join(path.read_text(encoding="utf-8") for path in python_files)
    assert "mcp.server.fastmcp" not in application_source
    assert "FastMCP" not in (PROJECT_ROOT / "wander" / "mcp_server.py").read_text(
        encoding="utf-8"
    )
    assert "from mcp.shared.exceptions import McpError" not in application_source

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert "mcp>=2,<3" in dependencies
    requirement_lines = (PROJECT_ROOT / "requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "mcp>=2,<3" in requirement_lines
    assert "mcp<2" not in requirement_lines
