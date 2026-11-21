import json
import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from .logging import log_node


logger = logging.getLogger(__name__)


class PlaceNote(BaseModel):
    number: int
    note: str | None = None


class SpecialistSummary(BaseModel):
    places: list[PlaceNote] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def tool_places(message: ToolMessage) -> list[Any] | None:
    """Read the existing MCP list result for summarization and diagnostics."""
    artifact = message.artifact
    if isinstance(artifact, dict):
        structured = artifact.get("structured_content")
        if isinstance(structured, dict) and isinstance(structured.get("result"), list):
            return structured["result"]

    content = message.content
    blocks = content if isinstance(content, list) else [content]
    places = []
    recognized = False
    for block in blocks:
        value = block.get("text") if isinstance(block, dict) else block
        if not isinstance(value, str):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and isinstance(decoded.get("result"), list):
            decoded = decoded["result"]
        if isinstance(decoded, list):
            recognized = True
            places.extend(decoded)
        elif isinstance(decoded, dict) and isinstance(decoded.get("name"), str):
            recognized = True
            places.append(decoded)
    return places if recognized else None


def _place_details(place: dict[str, Any]) -> dict[str, Any]:
    """Keep fields useful for the report and exclude bulky provider payloads."""
    selected = {
        key: place[key]
        for key in (
            "id", "name", "name_en", "category", "area", "location", "address", "coordinates",
            "description", "tags", "price", "estimated_duration_hours",
            "nightly_rate", "nightly_price", "currency", "availability_verified", "provider",
        )
        if key in place
    }
    metadata = place.get("provider_metadata")
    datasource = metadata.get("datasource") if isinstance(metadata, dict) else None
    raw = datasource.get("raw") if isinstance(datasource, dict) else None
    if isinstance(raw, dict):
        english_name = raw.get("name:en")
        if isinstance(english_name, str) and english_name.strip():
            selected["name_en"] = english_name
    if isinstance(selected.get("description"), str):
        selected["description"] = selected["description"][:450]
    return selected


def distinct_places(places: list[Any]) -> list[dict[str, Any]]:
    distinct = []
    names: set[str] = set()
    ids: set[str] = set()
    for place in places:
        name = place.get("name") if isinstance(place, dict) else None
        if not isinstance(name, str) or not name.strip():
            continue
        name_key = " ".join(name.casefold().split())
        place_id = place.get("id")
        if name_key in names or (isinstance(place_id, str) and place_id in ids):
            continue
        names.add(name_key)
        if isinstance(place_id, str) and place_id:
            ids.add(place_id)
        distinct.append(dict(place, name=name.strip()))
        if len(distinct) == 8:
            break
    return distinct


def build_specialist_graph(
    name: str, tools: list, model: Any, report_type: type[BaseModel], key: str, prompt: str,
):
    tool_node = ToolNode(tools)
    summary_model = model.with_structured_output(SpecialistSummary, method="json_schema")

    async def run_tools(state: MessagesState, config: RunnableConfig):
        return await tool_node.ainvoke(state, config=config)

    async def summarize(state: MessagesState):
        tool_message = state["messages"][-1]
        if isinstance(tool_message, ToolMessage) and tool_message.status != "error":
            raw = tool_places(tool_message) or []
            places = distinct_places([_place_details(place) for place in raw if isinstance(place, dict)])
            if not places:
                return {}
            numbered = {index: place for index, place in enumerate(places, start=1)}
            try:
                summary = await summary_model.ainvoke([
                    SystemMessage(content=prompt),
                    state["messages"][0],
                    HumanMessage(content=json.dumps(
                        [{"number": number, **place} for number, place in numbered.items()],
                        ensure_ascii=False,
                    )),
                ])
            except (OutputParserException, ValidationError) as exc:
                logger.debug("%s relevance notes unavailable: %s", name, type(exc).__name__)
                summary = SpecialistSummary(limitations=["Relevance notes were unavailable."])
            notes = {item.number: item.note for item in summary.places if item.number in numbered}
            order = dict.fromkeys([*notes, *numbered])
            findings = []
            for number in order:
                place = numbered[number].copy()
                if key == "hotels" and isinstance(place.get("name_en"), str) and place["name_en"].strip():
                    place["name"] = place["name_en"]
                location = place.get("location") or place.get("area") or place.get("address")
                if isinstance(location, dict):
                    location = ", ".join(str(location[k]) for k in ("road", "suburb", "city") if location.get(k))
                place["location"] = location if isinstance(location, str) and location.strip() else None
                if key == "attractions" and isinstance(notes.get(number), str):
                    place["relevance"] = notes[number].strip()[:180] or None
                findings.append(place)
            report = report_type.model_validate({key: findings, "limitations": summary.limitations})
            return {"messages": [AIMessage(content=report.model_dump_json())]}
        return {}

    graph = StateGraph(MessagesState)
    graph.add_node("tools", log_node(f"{name}.tools", run_tools))
    graph.add_node("summarize", log_node(f"{name}.summarize", summarize))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
