"""The validator catches implausible results."""

import json

import geopandas as gpd
from shapely.geometry import Polygon, box

from src.agent.validator import validate_results, validate_tool_result


def write_buildings(path, heights, geometry=None):
    """Write a GeoJSON with the same columns as extract_buildings produces."""
    n = len(heights)
    geometry = geometry or [box(296000, 5039000 + i * 30, 296020, 5039020 + i * 30) for i in range(n)]
    buildings = gpd.GeoDataFrame(
        {
            "building_id": [f"B{i + 1:03d}" for i in range(n)],
            "height_m": heights,
            "max_height_m": [h + 2 for h in heights],
            "area_m2": [400.0] * n,
            "point_count": [12000] * n,
            "purity": [0.8] * n,
        },
        geometry=geometry,
        crs="EPSG:2950",
    )
    centroids = buildings.geometry.centroid.to_crs("EPSG:4326")
    buildings["lon"], buildings["lat"] = centroids.x, centroids.y
    buildings.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    return str(path)


def test_648_m_building_is_rejected(tmp_path):
    path = write_buildings(tmp_path / "b.geojson", [27.4, 648.0])
    report = validate_results([("extract_buildings", json.dumps({"buildings_path": path}), False)])

    assert not report.is_valid
    assert "B002" in report.feedback(can_retry=True)
    assert "normalize_height" in report.feedback(can_retry=True)  # suggested fix


def test_plausible_buildings_pass(tmp_path):
    path = write_buildings(tmp_path / "b.geojson", [27.4, 31.2, 42.7])
    report = validate_results([("filter_buildings", json.dumps({"buildings_path": path}), False)])
    assert report.is_valid and not report.issues


def test_invalid_geometry_is_rejected(tmp_path):
    bow_tie = Polygon([(296000, 5039000), (296010, 5039010), (296010, 5039000), (296000, 5039010)])
    path = write_buildings(tmp_path / "b.geojson", [25.0], geometry=[bow_tie])
    report = validate_results([("extract_buildings", json.dumps({"buildings_path": path}), False)])
    assert any(i.rule == "geometry_valid" for i in report.errors)


def test_point_cloud_rules():
    assert validate_tool_result("load_point_cloud", {"point_count": 0, "crs": "EPSG:2950"})[0].rule == "empty_point_cloud"
    assert validate_tool_result("load_point_cloud", {"point_count": 10, "crs": None})[0].rule == "crs_defined"
    bad_hag = {"height_above_ground_m": {"p1": -0.1, "p99": 230.0}}
    assert validate_tool_result("normalize_height", bad_hag)[0].rule == "height_above_ground_range"


def test_class_percentages():
    good = {"class_percentages": {"ground": 50.0, "vegetation": 30.0, "building": 15.0, "other": 5.0}}
    bad = {"class_percentages": {"ground": 80.0, "vegetation": 30.0, "building": 15.0, "other": 5.0}}
    assert validate_tool_result("classify_lidar_points", good) == []
    assert validate_tool_result("classify_lidar_points", bad)[0].severity == "error"


def test_tool_errors_are_not_validated():
    report = validate_results([("load_point_cloud", json.dumps({"error": "boom"}), True)])
    assert report.is_valid
