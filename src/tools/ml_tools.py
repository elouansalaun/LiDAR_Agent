"""Agent tool for point classification (Random Forest today, swappable later)."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field

from src import config
from src.lidar import io
from src.ml import classify
from src.tools.files import derived_path, is_up_to_date, relative, resolve_input
from src.tools.registry import ToolError, tool


@tool
def classify_lidar_points(
    point_cloud_path: Annotated[str, Field(description="Path returned by normalize_height (must contain height above ground)")],
) -> dict:
    """Classify every LiDAR point into ground, vegetation, building or other with the trained Random Forest.
    Returns the class percentages and the path of the classified point cloud, to pass to extract_buildings."""
    source = resolve_input(point_cloud_path)
    output = derived_path(source, "classified")

    # Classification is the slowest step: reuse the result if nothing changed.
    reused = is_up_to_date(output, source, config.MODEL_PATH)
    if reused:
        cloud = io.read_point_cloud(output)
    else:
        cloud = io.read_point_cloud(source)
        if "height_above_ground" not in cloud:
            raise ToolError("This point cloud has no height above ground. Call normalize_height on it first.")
        classify.classify_lidar_points(cloud)
        io.save_point_cloud(cloud, output, extra_fields=["height_above_ground", "predicted_class"])

    model_info = {}
    if config.MODEL_METRICS_PATH.exists():
        metrics = json.loads(config.MODEL_METRICS_PATH.read_text())
        selected = metrics[metrics["selected"]]
        model_info = {"test_macro_f1": selected["macro_f1"], "test_mean_iou": selected["mean_iou"],
                      "test_building_f1": selected["per_class"]["building"]["f1"]}

    return {
        "point_cloud_path": relative(output),
        "point_count": len(cloud),
        "class_percentages": classify.class_percentages(cloud["predicted_class"]),
        "model": "RandomForest",
        "model_test_scores": model_info,
        "reused_existing_file": reused,
    }
