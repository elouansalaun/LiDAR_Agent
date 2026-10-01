"""Offline demo planner: a rule-based stand-in for Claude.

Lets you demo the agent without an API key. It plugs into the graph exactly
where Claude does (same `llm(system, messages, tools)` interface, same
Anthropic-shaped responses), so the tools, the validator and the UI run for
real. Only the "brain" is replaced:

    Claude                                  OfflinePlanner
    ---------------------------------       --------------------------------------
    understands any phrasing                regular expressions (heights, places, sizes)
    knows where landmarks are               a 4-entry gazetteer (src/places.py)
    plans the tool calls itself             a fixed plan, resumed from the history
    reacts to errors / validator freely     two scripted recoveries
    writes the answer                       a template filled from tool results

It answers building-height questions ("taller than 20 m", "between 30 and
60 m", "more than 10 floors", "the tallest buildings", follow-ups like "and
above 40 m?") and coverage questions. Anything else gets an honest
"not supported offline".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src import config
from src.places import DEFAULT_PLACE, Place, find_place, normalize_text

LABEL = "Offline demo planner (rule-based, no LLM)"
METRES_PER_FLOOR = 3.0

# --- Response objects shaped like the Anthropic SDK's ------------------------


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class PlannerResponse:
    content: list
    stop_reason: str  # "tool_use" or "end_turn", like a Claude message


# --- Understanding the question -----------------------------------------------

NUMBER = r"(\d+(?:[.,]\d+)?)"
METRES = r"(?:m|meters?|metres?)\b"

SQUARE_SIZE = rf"\b{NUMBER}\s*{METRES}\s*(?:x\s*\d+\s*{METRES}\s*)?(?:square|area|box|block|zone)\b"  # "a 300 m square"
RADIUS = rf"\bwithin\s+{NUMBER}\s*{METRES}\s+(?:of|around|from)\b"                                # "within 200 m of"
BETWEEN = rf"\bbetween\s+{NUMBER}\s*(?:{METRES})?\s*and\s+{NUMBER}"
AT_LEAST = rf"(?:\b(?:taller|higher|greater|bigger|more)\s+than|\babove|\bover|\bat\s+least|\bexceeding|>=?)\s*{NUMBER}"
AT_MOST = rf"(?:\b(?:shorter|lower|smaller|less)\s+than|\bbelow|\bunder|\bat\s+most|<=?)\s*{NUMBER}"
FLOORS = r"\b(?:floors?|storeys?|stories)\b"
TILE = r"\b23_\d{7}f08_dc\b"
COORDINATES = r"(-?\d{2}\.\d+)\s*,\s*(-?\d{2}\.\d+)"

BUILDING_NOUNS = r"\b(?:buildings?|towers?|roofs?|skyscrapers?|structures?|houses?)\b"
HEIGHT_WORDS = r"\b(?:tall|taller|tallest|high|higher|highest|heights?)\b"
COVERAGE_WORDS = r"\b(?:coverage|cover|covers|covered|tiles?|extent|dataset|data)\b"
OTHER_TOPICS = r"\b(?:trees?|vegetation|canopy|forest|slopes?|terrain|roads?|cars?|ground)\b"


@dataclass
class Intent:
    kind: str                      # "buildings", "coverage" or "unsupported"
    min_height: float = 0.0
    max_height: float | None = None
    place: Place | None = None     # named place
    center: tuple[float, float] | None = None  # explicit (lon, lat)
    size_m: float = 500.0
    tile_name: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_area(self) -> bool:
        return bool(self.place or self.center or self.tile_name)

    @property
    def has_filter(self) -> bool:
        return self.min_height > 0 or self.max_height is not None


def _number(text: str) -> float:
    return float(text.replace(",", "."))


def parse_question(question: str) -> Intent:
    text = normalize_text(question)
    intent = Intent(kind="unsupported")

    # 1. Area size, removed from the text so "500 m square" is not read as a height.
    for pattern, factor in ((SQUARE_SIZE, 1), (RADIUS, 2)):
        if match := re.search(pattern, text):
            size = _number(match.group(1)) * factor
            if size > config.MAX_AREA_KM2 ** 0.5 * 1000:
                intent.notes.append("The area was reduced to 1000 m × 1000 m, the limit for one request.")
            intent.size_m = min(max(size, 50.0), 1000.0)
            text = text[:match.start()] + " " + text[match.end():]

    # 2. Where: tile name, coordinates or a known place.
    if match := re.search(TILE, text):
        intent.tile_name = match.group(0).upper()
    elif match := re.search(COORDINATES, text):
        a, b = _number(match.group(1)), _number(match.group(2))
        intent.center = (a, b) if a < 0 else (b, a)  # longitudes are negative in Montreal
        text = text.replace(match.group(0), " ")
    else:
        intent.place = find_place(text)

    # 3. Height condition.
    if match := re.search(BETWEEN, text):
        intent.min_height, intent.max_height = sorted((_number(match.group(1)), _number(match.group(2))))
    else:
        if match := re.search(AT_LEAST, text):
            intent.min_height = _number(match.group(1))
        if match := re.search(AT_MOST, text):
            intent.max_height = _number(match.group(1))
    if re.search(FLOORS, text) and intent.has_filter:
        intent.min_height *= METRES_PER_FLOOR
        intent.max_height = intent.max_height * METRES_PER_FLOOR if intent.max_height is not None else None
        intent.notes.append(f"Floors were converted to heights assuming {METRES_PER_FLOOR:g} m per floor.")

    # 4. What kind of question is it?
    mentions_building = bool(re.search(BUILDING_NOUNS, text))
    mentions_other = bool(re.search(OTHER_TOPICS, text))
    if mentions_building or (not mentions_other and (intent.has_filter or re.search(HEIGHT_WORDS, text))):
        intent.kind = "buildings"   # includes follow-ups such as "and above 40 m?"
    elif not mentions_other and re.search(COVERAGE_WORDS, text):
        intent.kind = "coverage"
    return intent


# --- Reading the conversation ---------------------------------------------------


@dataclass
class ToolOutcome:
    name: str
    input: dict
    result: dict
    is_error: bool


def _current_question(messages: list) -> int:
    """Index of the latest user question (a plain-text user message)."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user" and isinstance(messages[index]["content"], str):
            return index
    raise ValueError("No user question in the conversation")


def _tool_outcomes(messages: list) -> list[ToolOutcome]:
    """Pair every tool call with its result, in order."""
    calls, outcomes = {}, []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if getattr(block, "type", None) == "tool_use":
                calls[block.id] = block
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                call = calls[block["tool_use_id"]]
                outcomes.append(ToolOutcome(call.name, call.input, json.loads(block["content"]), block["is_error"]))
    return outcomes


def _validator_texts(messages: list) -> list[str]:
    """Validator feedback blocks, in order."""
    return [
        block["text"]
        for message in messages if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"] if isinstance(block, dict) and block.get("type") == "text"
    ]


def _last(outcomes: list[ToolOutcome], name: str) -> ToolOutcome | None:
    return next((o for o in reversed(outcomes) if o.name == name and not o.is_error), None)


# --- The planner -------------------------------------------------------------------


class OfflinePlanner:
    """Callable with the same signature as ClaudeLLM: (system, messages, tools) -> response."""

    label = LABEL

    def __init__(self):
        self.calls = 0

    def __call__(self, system: str, messages: list, tools: list[dict]) -> PlannerResponse:
        self.calls += 1
        question_index = _current_question(messages)
        intent = parse_question(messages[question_index]["content"])
        turn = messages[question_index + 1:]
        earlier = _tool_outcomes(messages[:question_index])
        now = _tool_outcomes(turn)
        feedback = _validator_texts(turn[-1:])  # feedback attached to the latest tool results

        if intent.kind == "unsupported":
            return self._answer(UNSUPPORTED)
        if intent.kind == "coverage":
            return self._coverage(now)

        # Recovery: a failed tool or a rejected result.
        if now and now[-1].is_error:
            return self._answer(f"The `{now[-1].name}` step failed: {now[-1].result.get('error')}\n\n"
                                "Try another area or threshold.")
        if any("No retries left" in text for text in feedback):
            return self._answer("Some results still failed validation after retrying, so I am not reporting them "
                                "as correct:\n\n```\n" + "\n".join(feedback) + "\n```")
        if any(text.startswith("VALIDATOR: suspicious") for text in feedback):
            return self._recover(now[-1], intent)

        return self._next_building_step(intent, earlier, now, _validator_texts(turn))

    # -- building questions --

    def _next_building_step(self, intent: Intent, earlier: list, now: list, all_feedback: list[str]) -> PlannerResponse:
        done = {o.name for o in now if not o.is_error}
        previous_buildings = _last(earlier, "extract_buildings")
        follow_up = not intent.has_area and previous_buildings is not None

        if follow_up:
            source = previous_buildings
        else:
            if "load_point_cloud" not in done:
                return self._tool("load_point_cloud", self._area_arguments(intent), f"Cropping {_area_phrase(intent)}.")
            if "normalize_height" not in done:
                return self._tool("normalize_height", {"point_cloud_path": _last(now, "load_point_cloud").result["point_cloud_path"]},
                                  "Converting altitudes into heights above the ground.")
            if "classify_lidar_points" not in done:
                return self._tool("classify_lidar_points", {"point_cloud_path": _last(now, "normalize_height").result["point_cloud_path"]},
                                  "Classifying every point as ground, vegetation, building or other.")
            if "extract_buildings" not in done:
                return self._tool("extract_buildings", {"point_cloud_path": _last(now, "classify_lidar_points").result["point_cloud_path"]},
                                  "Grouping building points into individual buildings and measuring them.")
            source = _last(now, "extract_buildings")

        if "filter_buildings" not in done:
            arguments = {"buildings_path": source.result["buildings_path"], "min_height": intent.min_height}
            if intent.max_height is not None:
                arguments["max_height"] = intent.max_height
            reuse = "Re-using the buildings from the previous question. " if follow_up else ""
            return self._tool("filter_buildings", arguments, f"{reuse}Keeping buildings {_height_phrase(intent)}.")

        selected = _last(now, "filter_buildings")
        if "create_map" not in done and selected.result["building_count"] > 0:
            return self._tool("create_map", {
                "buildings_path": selected.result["buildings_path"],
                "title": f"Buildings {_height_phrase(intent)}",
                "area_bbox_lonlat": source.result.get("area_bbox_lonlat"),
            }, "Drawing the map.")

        return self._answer(self._building_answer(intent, source, now, follow_up, all_feedback))

    def _recover(self, rejected: ToolOutcome, intent: Intent) -> PlannerResponse:
        """Apply the validator's suggested fix for the step that was rejected."""
        if rejected.name in ("extract_buildings", "filter_buildings"):
            limit = min(intent.max_height or config.MAX_BUILDING_HEIGHT_M, config.MAX_BUILDING_HEIGHT_M)
            return self._tool(
                "filter_buildings",
                {"buildings_path": rejected.result["buildings_path"], "min_height": intent.min_height, "max_height": limit},
                f"The validator flagged implausible heights: excluding buildings above {limit:g} m.",
            )
        if rejected.name == "normalize_height":
            method = "estimate" if rejected.input.get("method") == "classified" else "classified"
            return self._tool("normalize_height", {"point_cloud_path": rejected.input["point_cloud_path"], "method": method},
                              f"The validator rejected the ground model: retrying with method '{method}'.")
        return self._answer(f"The `{rejected.name}` result failed validation and I have no automatic fix for it.")

    # -- coverage questions --

    def _coverage(self, now: list[ToolOutcome]) -> PlannerResponse:
        tiles = _last(now, "find_lidar_tiles")
        if tiles is None:
            return self._tool("find_lidar_tiles", {}, "Looking up the available LiDAR tiles.")
        info = tiles.result
        min_lon, min_lat, max_lon, max_lat = info["coverage_bbox_lonlat"]
        total = sum(t["point_count"] for t in info["tiles"])
        lines = [
            f"The data covers **{len(info['tiles'])} tiles** ({total / 1e6:.0f} million points) between longitudes "
            f"{min_lon:.4f} and {max_lon:.4f} and latitudes {min_lat:.4f} and {max_lat:.4f} "
            f"(Côte-des-Neiges, Montreal), in {info['crs']}.",
            "",
            "| Tile | Points | Density |",
            "|---|--:|--:|",
            *[f"| {t['name']} | {t['point_count'] / 1e6:.1f} M | {t['density_pts_per_m2']} pts/m² |" for t in info["tiles"]],
            "",
            "Places I know inside it: Université de Montréal, Saint Joseph's Oratory, Côte-des-Neiges / Queen-Mary.",
        ]
        return self._answer("\n".join(lines))

    # -- helpers --

    def _area_arguments(self, intent: Intent) -> dict:
        if intent.tile_name:
            return {"tile_name": intent.tile_name}
        place = intent.place or DEFAULT_PLACE
        lon, lat = intent.center or (place.lon, place.lat)
        return {"center_lon": lon, "center_lat": lat, "size_m": intent.size_m}

    def _tool(self, name: str, arguments: dict, sentence: str) -> PlannerResponse:
        return PlannerResponse([TextBlock(sentence), ToolUseBlock(f"offline_{self.calls}", name, arguments)], "tool_use")

    def _answer(self, text: str) -> PlannerResponse:
        return PlannerResponse([TextBlock(f"{text}\n\n_{LABEL}: answer templated from the tool results._")], "end_turn")

    def _building_answer(self, intent: Intent, source: ToolOutcome, now: list, follow_up: bool, all_feedback: list[str]) -> str:
        selected = _last(now, "filter_buildings").result
        count = selected["building_count"]
        area = "the same area as before" if follow_up else _area_phrase(intent)

        if count == 0:
            lines = [f"**No building {_height_phrase(intent)}** was found in {area}."]
            if source.result.get("height_m") and intent.max_height is None:
                lines.append(f"The tallest building there is {source.result['height_m']['max']:.1f} m.")
        else:
            if intent.has_filter:
                verb = "building is" if count == 1 else "buildings are"
                lines = [f"**{count} {verb} {_height_phrase(intent)}** in {area}."]
            else:
                lines = [f"**{count} buildings** were found in {area}, tallest first."]
            lines += ["", "| Building | Height | Highest point | Footprint | Location (lat, lon) |", "|---|--:|--:|--:|---|"]
            lines += [f"| {b['id']} | {b['height_m']:.1f} m | {b['max_height_m']:.1f} m | {b['area_m2']:,.0f} m² | "
                      f"{b['lat']:.5f}, {b['lon']:.5f} |" for b in selected["buildings"][:15]]
            if count > 15:
                lines.append(f"\n…and {count - 15} more (see the table and the map).")
            lines += ["", f"Average height **{selected['height_m']['mean']:.1f} m**, maximum **{selected['height_m']['max']:.1f} m**."]
            towers = [b for b in selected["buildings"] if b["max_height_m"] - b["height_m"] > 10]
            if towers:
                lines.append("Much higher points (towers, domes, spires): "
                             + ", ".join(f"{b['id']} reaches {b['max_height_m']:.0f} m" for b in towers[:3]) + ".")
            if mapped := _last(now, "create_map"):
                lines.append(f"Map: `{mapped.result['map_path']}`")

        notes = list(intent.notes)
        loaded = _last(now, "load_point_cloud")
        if loaded and loaded.result.get("note"):
            notes.append(loaded.result["note"])
        if not follow_up and not intent.has_area:
            notes.append("No area was named, so I used the 500 m × 500 m square at the centre of the coverage.")
        if any(text.startswith("VALIDATOR: suspicious") for text in all_feedback):
            notes.append("The validator rejected some heights as implausible; those buildings were excluded.")
        for text in all_feedback:
            if "VALIDATOR warnings" in text:
                warnings = text.split("VALIDATOR warnings", 1)[1].splitlines()[1:]
                notes += [f"Validator warning: {w.lstrip('- ')}" for w in warnings if w.strip()]
        if notes:
            lines += ["", *[f"- {note}" for note in dict.fromkeys(notes)]]

        lines += ["", "_Heights are the 98th percentile of roof points above ground; adjoining buildings are merged into one footprint._"]
        return "\n".join(lines)


def _height_phrase(intent: Intent) -> str:
    if intent.max_height is not None and intent.min_height > 0:
        return f"between {intent.min_height:g} and {intent.max_height:g} m tall"
    if intent.max_height is not None:
        return f"at most {intent.max_height:g} m tall"
    if intent.min_height > 0:
        return f"taller than {intent.min_height:g} m"
    return "of any height"


def _area_phrase(intent: Intent) -> str:
    if intent.tile_name:
        return f"tile {intent.tile_name}"
    size = f"the {intent.size_m:g} m × {intent.size_m:g} m square"
    if intent.center:
        return f"{size} around ({intent.center[1]:.5f}, {intent.center[0]:.5f})"
    return f"{size} around {(intent.place or DEFAULT_PLACE).name}"


UNSUPPORTED = (
    "The offline demo planner only understands **building-height** and **data-coverage** questions, for example:\n\n"
    "- *Find buildings taller than 20 m around Saint Joseph's Oratory*\n"
    "- *Which buildings are between 30 and 60 m high near Université de Montréal?*\n"
    "- *What are the tallest buildings in a 300 m square around Côte-des-Neiges?*\n"
    "- *And above 40 m?* (follow-up on the same area)\n"
    "- *Which area does the data cover?*\n\n"
    "With an API key, Claude plans the tool calls itself and handles free-form questions."
)
