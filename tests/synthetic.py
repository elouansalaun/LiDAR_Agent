"""Synthetic point clouds with a known answer, for the unit tests."""

from __future__ import annotations

import numpy as np
from pyproj import CRS

from src import config
from src.lidar.point_cloud import PointCloud


def make_scene(
    size_m: float = 100.0,
    spacing_m: float = 0.4,
    ground_slope: float = 0.05,
    buildings: list[tuple[float, float, float, float]] = ((20, 20, 20, 15.0), (60, 50, 20, 30.0)),
    origin: tuple[float, float] = (296000.0, 5039000.0),
    seed: int = 0,
) -> PointCloud:
    """Sloping ground + box-shaped buildings.

    buildings: (x, y, side, height) in metres relative to the scene corner.
    Ground points are classified ASPRS 2, roof points ASPRS 6 and
    predicted_class is set to the true class, so building analysis can be
    tested independently of the classifier.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(0, size_m, spacing_m), np.arange(0, size_m, spacing_m))
    x = xs.ravel() + rng.uniform(-0.05, 0.05, xs.size)
    y = ys.ravel() + rng.uniform(-0.05, 0.05, ys.size)
    ground_z = 150.0 + ground_slope * x          # altitude, like real data
    z = ground_z.copy()
    classification = np.full(x.size, config.ASPRS_GROUND, dtype=np.uint8)

    for bx, by, side, height in buildings:
        on_roof = (x >= bx) & (x < bx + side) & (y >= by) & (y < by + side)
        z[on_roof] = ground_z[on_roof] + height + rng.normal(0, 0.05, on_roof.sum())
        classification[on_roof] = config.ASPRS_BUILDING

    predicted = np.where(classification == config.ASPRS_BUILDING, config.BUILDING, config.GROUND).astype(np.uint8)
    n = x.size
    return PointCloud(
        data={
            "x": x + origin[0], "y": y + origin[1], "z": z,
            "intensity": rng.integers(20000, 50000, n).astype(np.uint16),
            "return_number": np.ones(n, dtype=np.uint8),
            "number_of_returns": np.ones(n, dtype=np.uint8),
            "classification": classification,
            "predicted_class": predicted,
        },
        crs=CRS.from_epsg(2950),
    )
