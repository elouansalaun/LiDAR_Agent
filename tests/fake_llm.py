"""A scripted stand-in for Claude, to test the agent graph without an API key.

It mimics the shape of an Anthropic response (content blocks + stop_reason)
and follows the standard building workflow, reading file paths from the
previous tool results like the real model would.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use(name, arguments, call_id):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


class ScriptedLLM:
    """Plays: load -> normalize -> classify -> extract -> filter -> map -> answer."""

    def __init__(self, area: dict, min_height: float = 20):
        self.area = area
        self.min_height = min_height
        self.calls = 0
        self.seen_messages = []

    def __call__(self, system, messages, tools):
        self.calls += 1
        self.seen_messages.append(messages[-1])
        last = messages[-1]["content"]
        results = [b for b in last if isinstance(b, dict) and b.get("type") == "tool_result"] if isinstance(last, list) else []
        result = json.loads(results[-1]["content"]) if results else {}
        feedback = [b["text"] for b in last if isinstance(b, dict) and b.get("type") == "text"] if isinstance(last, list) else []

        step = {
            None: ("load_point_cloud", lambda r: self.area),
            "load": ("normalize_height", lambda r: {"point_cloud_path": r["point_cloud_path"]}),
            "normalize": ("classify_lidar_points", lambda r: {"point_cloud_path": r["point_cloud_path"]}),
            "classify": ("extract_buildings", lambda r: {"point_cloud_path": r["point_cloud_path"]}),
            "extract": ("filter_buildings", lambda r: {"buildings_path": r["buildings_path"], "min_height": self.min_height}),
            "filter": ("create_map", lambda r: {"buildings_path": r["buildings_path"]}),
        }
        stage = self._stage(result)
        if stage == "map" or "error" in result:
            answer = f"Done. Last result: {json.dumps(result)[:200]}"
            if feedback:
                answer += "\nValidator said: " + feedback[0]
            return SimpleNamespace(content=[text_block(answer)], stop_reason="end_turn")

        name, make_arguments = step[stage]
        return SimpleNamespace(
            content=[text_block(f"Calling {name}."), tool_use(name, make_arguments(result), f"call_{self.calls}")],
            stop_reason="tool_use",
        )

    @staticmethod
    def _stage(result: dict):
        if not result:
            return None
        if "map_path" in result:
            return "map"
        if "filter" in result:
            return "filter"
        if "buildings_path" in result:
            return "extract"
        if "class_percentages" in result:
            return "classify"
        if "height_above_ground_m" in result:
            return "normalize"
        return "load"
