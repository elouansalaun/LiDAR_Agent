"""Ground model (DTM) and height normalisation.

"Height above ground" (HAG) is the single most useful LiDAR feature: a roof at
Z = 210 m on the slope of Mount Royal and a roof at Z = 130 m downtown can both
be 25 m tall buildings. We compute it in three steps:

1. pick the ground points (ASPRS class 2 when the file is classified,
   otherwise a simple morphological estimate),
2. rasterise them into a 1 m digital terrain model (DTM), filling holes
   under buildings with the nearest ground value,
3. for every point, HAG = Z - DTM(x, y), with bilinear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from src import config
from src.lidar.point_cloud import PointCloud

# Minimum number of classified ground points before we trust the vendor classes.
MIN_GROUND_POINTS = 1000


@dataclass
class DTM:
    """A north-up elevation grid. Row 0 is the northern edge (like a GeoTIFF)."""

    elevation: np.ndarray  # (rows, cols), metres
    min_x: float
    max_y: float
    resolution: float
    crs: object = None

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the ground elevation at (x, y)."""
        # Continuous (row, col) positions, relative to cell centres.
        cols = (x - self.min_x) / self.resolution - 0.5
        rows = (self.max_y - y) / self.resolution - 0.5
        return ndimage.map_coordinates(self.elevation, [rows, cols], order=1, mode="nearest")

    def save(self, path: str | Path) -> Path:
        """Write the DTM as a GeoTIFF (handy to inspect in QGIS)."""
        import rasterio
        from rasterio.transform import from_origin

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path, "w", driver="GTiff",
            height=self.elevation.shape[0], width=self.elevation.shape[1],
            count=1, dtype="float32", crs=self.crs,
            transform=from_origin(self.min_x, self.max_y, self.resolution, self.resolution),
        ) as dst:
            dst.write(self.elevation.astype(np.float32), 1)
        return path


def rasterize(x, y, values, min_x, max_y, resolution, shape, reducer="mean") -> np.ndarray:
    """Aggregate point values into a grid. Empty cells are NaN.

    reducer: "mean", "min" or "max".
    """
    rows = ((max_y - y) / resolution).astype(int).clip(0, shape[0] - 1)
    cols = ((x - min_x) / resolution).astype(int).clip(0, shape[1] - 1)
    flat = rows * shape[1] + cols
    n_cells = shape[0] * shape[1]

    if reducer == "mean":
        sums = np.bincount(flat, weights=values, minlength=n_cells)
        counts = np.bincount(flat, minlength=n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = sums / counts
    elif reducer in ("min", "max"):
        fill = np.inf if reducer == "min" else -np.inf
        grid = np.full(n_cells, fill)
        ufunc = np.minimum if reducer == "min" else np.maximum
        ufunc.at(grid, flat, values)
        grid[np.isinf(grid)] = np.nan
    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    return grid.reshape(shape)


def fill_holes(grid: np.ndarray) -> np.ndarray:
    """Replace NaN cells with the value of the nearest valid cell."""
    missing = np.isnan(grid)
    if not missing.any():
        return grid
    if missing.all():
        raise ValueError("Cannot fill a grid that has no valid cell")
    _, (rows, cols) = ndimage.distance_transform_edt(missing, return_indices=True)
    return grid[rows, cols]


def grid_shape(cloud: PointCloud, resolution: float) -> tuple[float, float, tuple[int, int]]:
    min_x, min_y, max_x, max_y = cloud.bounds
    min_x, max_y = np.floor(min_x), np.ceil(max_y)
    cols = int(np.ceil((max_x - min_x) / resolution)) + 1
    rows = int(np.ceil((max_y - min_y) / resolution)) + 1
    return min_x, max_y, (rows, cols)


def build_dtm(cloud: PointCloud, ground_mask: np.ndarray, resolution: float = config.DTM_RESOLUTION_M) -> DTM:
    """Build a DTM covering the whole cloud from the points flagged as ground."""
    min_x, max_y, shape = grid_shape(cloud, resolution)
    grid = rasterize(
        cloud["x"][ground_mask], cloud["y"][ground_mask], cloud["z"][ground_mask],
        min_x, max_y, resolution, shape, reducer="mean",
    )
    return DTM(fill_holes(grid), min_x, max_y, resolution, cloud.crs)


def estimate_ground_mask(
    cloud: PointCloud,
    cell_size: float = 1.0,
    window_m: float = 40.0,
    tolerance_m: float = 0.5,
) -> np.ndarray:
    """Rough ground detection for unclassified files.

    Takes the lowest point of each cell, then applies a morphological opening
    (erosion followed by dilation) with a window larger than most buildings:
    anything "sticking up" and narrower than the window is removed, leaving an
    approximate terrain surface. Points within `tolerance_m` of that surface
    are flagged as ground.

    This is a simplified version of the classic Zhang et al. (2003) filter.
    Very large buildings (wider than `window_m`) are its known weakness; PDAL's
    SMRF filter is the production-grade alternative.
    """
    min_x, max_y, shape = grid_shape(cloud, cell_size)
    lowest = fill_holes(rasterize(cloud["x"], cloud["y"], cloud["z"], min_x, max_y, cell_size, shape, reducer="min"))
    window = max(3, int(window_m / cell_size))
    terrain = ndimage.grey_opening(lowest, size=(window, window))

    surface = DTM(terrain, min_x, max_y, cell_size).sample(cloud["x"], cloud["y"])
    return np.abs(cloud["z"] - surface) < tolerance_m


def normalize_height(cloud: PointCloud, method: str = "auto") -> tuple[PointCloud, DTM, str]:
    """Add a `height_above_ground` attribute to the cloud.

    method:
        "classified" - use the ASPRS ground class (2) delivered with the data
        "estimate"   - use estimate_ground_mask() (for unclassified data)
        "auto"       - "classified" if enough ground points exist, else "estimate"

    Returns the cloud, the DTM and the method actually used.
    """
    classified_ground = cloud["classification"] == config.ASPRS_GROUND
    if method == "auto":
        method = "classified" if classified_ground.sum() >= MIN_GROUND_POINTS else "estimate"

    if method == "classified":
        ground_mask = classified_ground
    elif method == "estimate":
        ground_mask = estimate_ground_mask(cloud)
    else:
        raise ValueError(f"Unknown method '{method}' (use auto, classified or estimate)")

    dtm = build_dtm(cloud, ground_mask)
    hag = cloud["z"] - dtm.sample(cloud["x"], cloud["y"])
    cloud.add("height_above_ground", hag.astype(np.float32))
    return cloud, dtm, method
