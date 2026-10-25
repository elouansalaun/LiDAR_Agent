"""Deliverable: `classify_lidar_points(point_cloud)`.

The rest of the project (building analysis, agent tools) only talks to this
function, so the Random Forest can later be swapped for PointNet++ or any
other model without touching the callers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.lidar import features
from src.lidar.point_cloud import PointCloud


@lru_cache(maxsize=2)
def load_model(path: str = str(config.MODEL_PATH)) -> dict:
    """Load the model bundle saved by src/ml/train.py (cached after the first call)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m src.ml.train` first.")
    return joblib.load(path)


def classify_lidar_points(point_cloud: PointCloud, model_path: str | Path = config.MODEL_PATH, chunk_size: int = 2_000_000) -> PointCloud:
    """Predict ground / vegetation / building / other for every point.

    The cloud must have `height_above_ground`; missing features are computed
    on the fly. Adds a `predicted_class` attribute (uint8, see config.CLASS_NAMES).
    """
    bundle = load_model(str(model_path))
    model, feature_names = bundle["model"], bundle["features"]

    if any(name not in point_cloud for name in feature_names):
        needs_geometric = any(name in features.GEOMETRIC_FEATURES for name in feature_names)
        features.compute_features(point_cloud, geometric=needs_geometric)

    X = features.feature_matrix(point_cloud, feature_names)
    predictions = np.empty(len(X), dtype=np.uint8)
    for start in range(0, len(X), chunk_size):
        predictions[start:start + chunk_size] = model.predict(X[start:start + chunk_size])

    point_cloud.add("predicted_class", predictions)
    return point_cloud


def class_percentages(predicted_class: np.ndarray) -> dict[str, float]:
    counts = np.bincount(predicted_class, minlength=len(config.CLASS_NAMES))
    return {name: round(100 * counts[label] / max(len(predicted_class), 1), 2) for label, name in config.CLASS_NAMES.items()}
