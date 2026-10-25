"""Sanity checks on tool results before the agent may use them.

An LLM will happily report "Building B001 is 648 m tall" if a tool says so.
The validator sits between the tools and the agent and checks physical /
geometric plausibility with plain rules (no LLM involved):

    point_count > 0              CRS is defined
    0 < building_height < 150    building_area > 0
    0 <= percentage <= 100       geometry is valid

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd

from src import config

# Likely causes shown to the agent for each rule, so the retry is informed.
POSSIBLE_CAUSES = {
    "building_height_range": [
        "heights measured from sea level instead of the ground (normalize_height skipped or failed)",
        "vegetation or noise points classified as building",
        "incorrect coordinate / elevation handling",
    ],
    "height_above_ground_range": [
        "the ground model is wrong (too few ground points, or 'estimate' method on a very large building)",
        "the point cloud contains noise points far above or below the scene",
    ],
    "empty_point_cloud": ["the area is outside the LiDAR coverage (check find_lidar_tiles)"],
    "crs_defined": ["the file has no coordinate reference system; outputs cannot be placed on a map"],
    "percentages": ["a bug in the classification summary"],
    "building_area": ["degenerate footprints produced by the segmentation"],
    "geometry_valid": ["self-intersecting footprints produced by the vectorisation"],
}

SUGGESTED_FIXES = {
    "building_height_range": (
        "Check that the point cloud passed to classify_lidar_points came from normalize_height. "
        "If it did, call extract_buildings again with a lower height_percentile (e.g. 90), "
        f"or use filter_buildings with max_height={config.MAX_BUILDING_HEIGHT_M:g} and report the excluded buildings as suspicious."
    ),
    "height_above_ground_range": "Call normalize_height again with method='classified' (or 'estimate' if that fails).",
    "empty_point_cloud": "Call find_lidar_tiles and choose an area inside the coverage.",
    "crs_defined": "Report to the user that the data has no CRS.",
    "percentages": "Report the problem to the user.",
    "building_area": "Call extract_buildings again with a larger min_area_m2.",
    "geometry_valid": "Call extract_buildings again; if it persists, report it.",
}


@dataclass
class Issue:
    tool: str
    rule: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def feedback(self, can_retry: bool) -> str:
        """Message for the LLM describing what is wrong and what to do."""
        lines = []
        if self.errors:
            lines.append("VALIDATOR: suspicious results detected - do not report these numbers as-is.")
            for rule in dict.fromkeys(i.rule for i in self.errors):  # unique, in order
                for issue in (i for i in self.errors if i.rule == rule):
                    lines.append(f"- [{issue.tool}] {issue.message}")
                lines.append("  Possible causes: " + "; ".join(POSSIBLE_CAUSES.get(rule, ["unknown"])))
                lines.append("  Suggested fix: " + SUGGESTED_FIXES.get(rule, "Report the problem to the user."))
            if can_retry:
                lines.append("Fix the problem and retry.")
            else:
                lines.append("No retries left: stop, and tell the user which results failed validation and why.")
        if self.warnings:
            lines.append("VALIDATOR warnings (results accepted, mention them if relevant):")
            lines += [f"- [{i.tool}] {i.message}" for i in self.warnings]
        return "\n".join(lines)


def validate_tool_result(tool_name: str, result: dict) -> list[Issue]:
    """Apply the rules relevant to one tool's output."""
    issues: list[Issue] = []

    def error(rule, message):
        issues.append(Issue(tool_name, rule, message, "error"))

    def warning(rule, message):
        issues.append(Issue(tool_name, rule, message, "warning"))

    if "point_count" in result and result["point_count"] <= 0:
        error("empty_point_cloud", "the point cloud contains 0 points")

    if "crs" in result and not result["crs"]:
        error("crs_defined", "the coordinate reference system is not defined")

    if "height_above_ground_m" in result:
        hag = result["height_above_ground_m"]
        if hag["p1"] < -2:
            error("height_above_ground_range", f"1% of points are more than {-hag['p1']:.1f} m below the ground model")
        if hag["p99"] > config.MAX_BUILDING_HEIGHT_M:
            error("height_above_ground_range",
                  f"99th percentile of height above ground is {hag['p99']:.0f} m (> {config.MAX_BUILDING_HEIGHT_M:g} m)")

    if "class_percentages" in result:
        percentages = result["class_percentages"]
        if any(not 0 <= p <= 100 for p in percentages.values()) or abs(sum(percentages.values()) - 100) > 0.5:
            error("percentages", f"class percentages are not valid: {percentages}")
        elif percentages.get("ground", 0) < 5:
            warning("percentages", f"only {percentages.get('ground', 0)}% ground points: the area may be a single roof, "
                                   "or height normalisation may be off")

    if "buildings_path" in result:
        issues += [Issue(tool_name, i.rule, i.message, i.severity) for i in validate_buildings_file(result["buildings_path"])]

    return issues


def validate_buildings_file(path: str | Path) -> list[Issue]:
    """Check *every* building in the GeoJSON (tool results only list the tallest ones)."""
    path = Path(path)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    buildings = gpd.read_file(path)
    if len(buildings) == 0:
        return []

    issues = []
    if buildings.crs is None:
        issues.append(Issue("buildings", "crs_defined", "the buildings file has no CRS"))

    too_low = buildings[buildings["height_m"] <= 0]
    too_high = buildings[buildings["height_m"] >= config.MAX_BUILDING_HEIGHT_M]
    for row in too_high.itertuples():
        issues.append(Issue("buildings", "building_height_range",
                            f"building {row.building_id} height = {row.height_m:.1f} m (limit {config.MAX_BUILDING_HEIGHT_M:g} m)"))
    for row in too_low.itertuples():
        issues.append(Issue("buildings", "building_height_range", f"building {row.building_id} height = {row.height_m:.1f} m (must be > 0)"))

    for row in buildings[buildings["area_m2"] <= 0].itertuples():
        issues.append(Issue("buildings", "building_area", f"building {row.building_id} has area {row.area_m2} m²"))

    invalid = buildings[~buildings.geometry.is_valid]
    if len(invalid):
        issues.append(Issue("buildings", "geometry_valid", f"{len(invalid)} invalid footprint geometries ({', '.join(invalid['building_id'][:5])})"))

    # Plausible roof height but a single point far above it: antenna, crane or bird.
    antenna_like = (buildings["max_height_m"] >= config.MAX_BUILDING_HEIGHT_M) & (buildings["height_m"] < config.MAX_BUILDING_HEIGHT_M)
    for row in buildings[antenna_like].itertuples():
        issues.append(Issue("buildings", "building_height_range",
                            f"building {row.building_id} has a point at {row.max_height_m:.0f} m (roof at {row.height_m:.0f} m): "
                            "probably an antenna, crane or noise", "warning"))
    return issues


def validate_results(results: list[tuple[str, str, bool]]) -> ValidationReport:
    """Validate the outputs of one round of tool calls.

    `results` holds (tool_name, json_output, is_error). Tool errors are not
    validated: the agent already sees the error message.
    """
    report = ValidationReport()
    for tool_name, output, is_error in results:
        if is_error:
            continue
        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            continue
        report.issues += validate_tool_result(tool_name, result)
    return report
