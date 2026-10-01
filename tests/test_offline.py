"""The offline demo planner: question parsing, multi-turn sessions and validator recovery."""

import json

import pytest

from src import config
from src.agent.graph import LidarAgent
from src.agent.offline import OfflinePlanner, TextBlock, ToolUseBlock, parse_question
from tests.test_validator import write_buildings

HAS_DATA = config.MODEL_PATH.exists() and any(config.RAW_TILES_DIR.glob("*.[lL][aA][zZ]"))


@pytest.mark.parametrize("question, kind, min_height, max_height, place, size", [
    ("Find buildings taller than 20 meters in this LiDAR area.", "buildings", 20, None, "Centre of the coverage", 500),
    ("Which buildings are between 30 and 60 m high near Université de Montréal?", "buildings", 30, 60, "Université de Montréal", 500),
    ("What are the tallest buildings in a 300 m square around Côte-des-Neiges?", "buildings", 0, None, "Côte-des-Neiges / Queen-Mary", 300),
    ("Buildings with more than 10 floors within 200 m of the oratoire", "buildings", 30, None, "Saint Joseph's Oratory", 400),
    ("And above 40 m?", "buildings", 40, None, None, 500),
    ("Which area does the data cover?", "coverage", 0, None, None, 500),
    ("Find trees taller than 20 m", "unsupported", 20, None, None, 500),
    ("hello", "unsupported", 0, None, None, 500),
])
def test_parse_question(question, kind, min_height, max_height, place, size):
    intent = parse_question(question)
    assert intent.kind == kind
    assert intent.min_height == min_height and intent.max_height == max_height
    assert (intent.place.name if intent.place else None) == place
    assert intent.size_m == size


def test_coordinates_and_tiles():
    assert parse_question("buildings over 25 m at 45.5035, -73.6150").center == (-73.615, 45.5035)
    assert parse_question("buildings above 20 m in tile 23_2955039F08_DC").tile_name == "23_2955039F08_DC"


def tool_names(result):
    return [e["name"] for e in result.trace if e["type"] == "tool_call"]


def test_validator_rejection_is_recovered(tmp_path):
    """A previous turn left a buildings file with a 648 m 'building'; the follow-up must not report it."""
    path = write_buildings(tmp_path / "b.geojson", [27.4, 31.2, 648.0])
    agent = LidarAgent(llm=OfflinePlanner())
    agent.messages = [  # what an earlier extract_buildings turn looks like in the history
        {"role": "user", "content": "Find buildings around the oratory"},
        {"role": "assistant", "content": [ToolUseBlock("t1", "extract_buildings", {"point_cloud_path": "x.laz"})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": False,
                                      "content": json.dumps({"buildings_path": path, "height_m": {"max": 648.0}})}]},
        {"role": "assistant", "content": [TextBlock("Done.")]},
    ]
    result = agent.ask("And above 20 m?")

    assert tool_names(result) == ["filter_buildings", "filter_buildings", "create_map"]
    rejections = [e for e in result.trace if e["type"] == "validation" and not e["valid"]]
    assert len(rejections) == 1
    assert "**2 buildings are taller than 20 m**" in result.answer
    assert "648" not in result.answer
    assert "rejected some heights" in result.answer


def test_unsupported_question_calls_no_tool():
    result = LidarAgent(llm=OfflinePlanner()).ask("Find areas with slopes greater than 30°")
    assert tool_names(result) == []
    assert "only understands" in result.answer


@pytest.mark.skipif(not HAS_DATA, reason="needs the raw tiles and a trained model")
def test_offline_session_on_real_data():
    agent = LidarAgent(llm=OfflinePlanner())

    first = agent.ask("Find buildings taller than 10 m in a 150 m square around Université de Montréal")
    assert tool_names(first) == ["load_point_cloud", "normalize_height", "classify_lidar_points",
                                 "extract_buildings", "filter_buildings", "create_map"]
    assert "taller than 10 m" in first.answer

    follow_up = agent.ask("and above 20 m?")
    assert tool_names(follow_up)[0] == "filter_buildings"  # re-uses the extracted buildings
    assert "load_point_cloud" not in tool_names(follow_up)

    coverage = agent.ask("Which area does the data cover?")
    assert tool_names(coverage) == ["find_lidar_tiles"] and "4 tiles" in coverage.answer
