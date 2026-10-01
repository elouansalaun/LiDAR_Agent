"""Tool layer and agent graph, with a fake LLM (no API key needed)."""

import json
from types import SimpleNamespace

import pytest

from src import config
from src.agent.graph import LidarAgent
from src.tools import TOOLS, tool_schemas
from tests.fake_llm import ScriptedLLM, text_block, tool_use
from tests.test_validator import write_buildings


def test_every_tool_has_a_description_and_schema():
    schemas = tool_schemas()
    assert [s["name"] for s in schemas] == sorted(TOOLS)
    for schema in schemas:
        assert len(schema["description"]) > 30
        assert schema["input_schema"]["type"] == "object"


def test_invalid_arguments_are_reported_not_raised():
    output, is_error = TOOLS["filter_buildings"].run_json({"min_height": -1})
    message = json.loads(output)["error"]
    assert is_error
    assert "buildings_path" in message and "min_height" in message


def test_missing_file_is_reported():
    output, is_error = TOOLS["classify_lidar_points"].run_json({"point_cloud_path": "does_not_exist.laz"})
    assert is_error and "not found" in json.loads(output)["error"]


class RetryingLLM:
    """First filters a file containing a 648 m 'building'; after validator feedback, excludes it."""

    def __init__(self, path):
        self.path = path
        self.messages_seen = []

    def __call__(self, system, messages, tools):
        self.messages_seen.append(messages[-1])
        n = len(self.messages_seen)
        if n == 1:
            return SimpleNamespace(content=[tool_use("filter_buildings", {"buildings_path": self.path, "min_height": 20}, "c1")],
                                   stop_reason="tool_use")
        if n == 2:
            return SimpleNamespace(content=[tool_use("filter_buildings", {"buildings_path": self.path, "min_height": 20,
                                                                          "max_height": 150}, "c2")],
                                   stop_reason="tool_use")
        return SimpleNamespace(content=[text_block("2 buildings are taller than 20 m.")], stop_reason="end_turn")


def test_validator_feedback_reaches_the_llm_and_agent_retries(tmp_path):
    path = write_buildings(tmp_path / "b.geojson", [27.4, 31.2, 648.0])
    llm = RetryingLLM(path)
    result = LidarAgent(llm=llm).ask("Find buildings taller than 20 m")

    # The message sent to the LLM after the first tool call contains the tool result + validator text.
    feedback_message = llm.messages_seen[1]["content"]
    assert feedback_message[0]["type"] == "tool_result"
    assert feedback_message[-1]["text"].startswith("VALIDATOR")
    assert "B003" in feedback_message[-1]["text"]

    validations = [e for e in result.trace if e["type"] == "validation"]
    assert len(validations) == 1 and not validations[0]["valid"]
    assert result.answer == "2 buildings are taller than 20 m."


class StubbornLLM:
    """Keeps calling the same failing tool, to test the retry and step limits."""

    def __init__(self, path):
        self.path = path
        self.calls = 0

    def __call__(self, system, messages, tools):
        self.calls += 1
        last = messages[-1]["content"]
        if isinstance(last, list) and "No retries left" in str(last[-1]):
            return SimpleNamespace(content=[text_block("Results failed validation.")], stop_reason="end_turn")
        return SimpleNamespace(content=[tool_use("filter_buildings", {"buildings_path": self.path}, f"c{self.calls}")],
                               stop_reason="tool_use")


def test_validation_retries_are_limited(tmp_path):
    path = write_buildings(tmp_path / "b.geojson", [648.0])
    result = LidarAgent(llm=StubbornLLM(path)).ask("Find buildings")

    rejections = [e for e in result.trace if e["type"] == "validation"]
    assert len(rejections) == config.MAX_VALIDATION_RETRIES + 1
    assert "No retries left" in rejections[-1]["feedback"]
    assert result.answer == "Results failed validation."


def test_step_limit(tmp_path):
    path = write_buildings(tmp_path / "b.geojson", [30.0])
    result = LidarAgent(llm=StubbornLLM(path), max_steps=3).ask("Find buildings")
    assert result.steps == 3
    assert result.answer.startswith("Stopped after 3 steps")


@pytest.mark.skipif(not (config.MODEL_PATH.exists() and any(config.RAW_TILES_DIR.glob("*.[lL][aA][zZ]"))),
                    reason="needs the raw tiles and a trained model")
def test_end_to_end_on_real_data():
    """Full workflow on a small real area, driven through the graph."""
    llm = ScriptedLLM({"center_lon": -73.6150, "center_lat": 45.5035, "size_m": 150}, min_height=10)
    agent = LidarAgent(llm=llm)
    result = agent.ask("Find buildings taller than 10 m around Université de Montréal")

    tool_names = [e["name"] for e in result.trace if e["type"] == "tool_call"]
    assert tool_names == ["load_point_cloud", "normalize_height", "classify_lidar_points",
                          "extract_buildings", "filter_buildings", "create_map"]
    assert not any(e["is_error"] for e in result.trace if e["type"] == "tool_call")
    assert (config.PROJECT_ROOT / result.artifacts["map_path"]).exists()
