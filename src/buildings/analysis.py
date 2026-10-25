"""From classified points to individual buildings.

    classified points -> building points -> 2D building mask (1 m grid)
    -> clean-up -> connected components -> polygons + height/area/centroid
    -> drop small or "impure" components

Why a 2D grid instead of 3D clustering? Seen from above, a building is a
connected patch of roof. Rasterising is fast (millions of points in a
second), easy to reason about, and gives polygons for free. The trade-off:
buildings that touch (e.g. Montreal row houses) are merged into one "building
block".
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio import features as rio_features
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from src import config
from src.lidar.ground import grid_shape
from src.lidar.point_cloud import PointCloud


def is_building(cloud: PointCloud, source: str = "predicted") -> np.ndarray:
    """Boolean mask of the points labelled as building.

    source: "predicted" uses our classifier output (`predicted_class`),
            "vendor" uses the ASPRS class 6 delivered with the data (useful to
            compare our results against the reference classification).
    """
    if source == "predicted":
        if "predicted_class" not in cloud:
            raise ValueError("The cloud has no `predicted_class`. Run classify_lidar_points() first.")
        return cloud["predicted_class"] == config.BUILDING
    if source == "vendor":
        return cloud["classification"] == config.ASPRS_BUILDING
    raise ValueError(f"Unknown source '{source}' (use 'predicted' or 'vendor')")


def extract_building_points(cloud: PointCloud, source: str = "predicted") -> PointCloud:
    return cloud.subset(is_building(cloud, source))


def cell_index(x: np.ndarray, y: np.ndarray, grid_origin: tuple[float, float], shape: tuple[int, int], resolution: float):
    """(row, col) of the grid cell containing each point. Row 0 is the northern edge."""
    min_x, max_y = grid_origin
    rows = ((max_y - y) / resolution).astype(int).clip(0, shape[0] - 1)
    cols = ((x - min_x) / resolution).astype(int).clip(0, shape[1] - 1)
    return rows, cols


def building_mask(points: PointCloud, grid_origin: tuple[float, float], shape: tuple[int, int], resolution: float) -> np.ndarray:
    """Boolean raster: True where at least a couple of building points fall in the cell."""
    rows, cols = cell_index(points["x"], points["y"], grid_origin, shape, resolution)
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, (rows, cols), 1)
    mask = counts >= 2

    # Clean-up: the opening removes isolated misclassified cells (a tree crown
    # predicted as building here and there), the closing fills small gaps in roofs.
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    return mask


def segment_buildings(
    cloud: PointCloud,
    source: str = "predicted",
    resolution: float = config.BUILDING_GRID_M,
    min_area_m2: float = config.BUILDING_MIN_AREA_M2,
    height_percentile: float = config.BUILDING_HEIGHT_PERCENTILE,
    min_purity: float = config.BUILDING_MIN_PURITY,
) -> gpd.GeoDataFrame:
    """Find every building in the cloud and compute its statistics.

    Returns a GeoDataFrame (one row per building, CRS of the cloud) with:
        height_m      robust roof height (percentile of height above ground)
        max_height_m  highest building point (antennas, spires...)
        area_m2       footprint area
        point_count   number of building points
        purity        share of building points among all points > 2 m in the footprint
        centroid_x/y  in the cloud CRS, lon/lat in WGS84

    `purity` removes a typical classifier error: a few points of a tree crown
    predicted as building form a small, tall blob. Inside a real footprint
    most elevated points are building points (purity 0.65-0.9 on our data);
    inside such a blob most are still vegetation (purity 0.1-0.25).
    """
    if "height_above_ground" not in cloud:
        raise ValueError("The cloud has no `height_above_ground`. Run normalize_height() first.")

    building_points = is_building(cloud, source)
    points = cloud.subset(building_points)
    columns = ["building_id", "height_m", "max_height_m", "area_m2", "point_count", "purity",
               "centroid_x", "centroid_y", "lon", "lat", "geometry"]
    if len(points) == 0:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=cloud.crs)

    # Same grid for the whole cloud so results do not depend on where buildings are.
    min_x, max_y, grid = grid_shape(cloud, resolution)
    mask = building_mask(points, (min_x, max_y), grid, resolution)
    labels, n_labels = ndimage.label(mask)  # 4-connectivity: diagonal contact does not merge

    # Which component does every building point belong to? (0 = none)
    point_labels = labels[cell_index(points["x"], points["y"], (min_x, max_y), grid, resolution)]
    purity = _purity_by_label(cloud, building_points, labels, (min_x, max_y), resolution)

    transform = from_origin(min_x, max_y, resolution, resolution)
    polygons = _polygons_by_label(labels, transform)
    hag = points["height_above_ground"]

    records = []
    for label in range(1, n_labels + 1):
        polygon = polygons.get(label)
        if polygon is None or polygon.area < min_area_m2 or purity[label] < min_purity:
            continue
        heights = hag[point_labels == label]
        if len(heights) == 0:
            continue
        records.append({
            "height_m": float(np.percentile(heights, height_percentile)),
            "max_height_m": float(heights.max()),
            "area_m2": float(polygon.area),
            "point_count": int(len(heights)),
            "purity": round(float(purity[label]), 2),
            "geometry": polygon,
        })

    if not records:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=cloud.crs)

    buildings = gpd.GeoDataFrame(records, geometry="geometry", crs=cloud.crs)
    buildings = buildings.sort_values("height_m", ascending=False).reset_index(drop=True)
    buildings.insert(0, "building_id", [f"B{i + 1:03d}" for i in range(len(buildings))])

    centroids = buildings.geometry.centroid
    buildings["centroid_x"], buildings["centroid_y"] = centroids.x, centroids.y
    centroids_wgs84 = centroids.to_crs("EPSG:4326")
    buildings["lon"], buildings["lat"] = centroids_wgs84.x, centroids_wgs84.y

    for column in ("height_m", "max_height_m", "area_m2", "centroid_x", "centroid_y"):
        buildings[column] = buildings[column].round(2)
    buildings[["lon", "lat"]] = buildings[["lon", "lat"]].round(6)
    return buildings[columns]


def _purity_by_label(cloud: PointCloud, building_points: np.ndarray, labels: np.ndarray, grid_origin, resolution: float) -> np.ndarray:
    """For each component: building points / all points higher than 2 m, inside its cells."""
    elevated = cloud["height_above_ground"] > 2.0
    rows, cols = cell_index(cloud["x"][elevated], cloud["y"][elevated], grid_origin, labels.shape, resolution)
    point_labels = labels[rows, cols]

    n_bins = labels.max() + 1
    total = np.bincount(point_labels, minlength=n_bins)
    building = np.bincount(point_labels, weights=building_points[elevated], minlength=n_bins)
    return building / np.maximum(total, 1)


def _polygons_by_label(labels: np.ndarray, transform) -> dict:
    """Vectorise a label raster into one (multi)polygon per label."""
    pieces: dict[int, list] = {}
    for geometry, value in rio_features.shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
        pieces.setdefault(int(value), []).append(shape(geometry))
    # Light simplification removes the 1 m "staircase" edges of the raster.
    return {label: unary_union(parts).simplify(0.5) for label, parts in pieces.items()}


def filter_by_height(buildings: gpd.GeoDataFrame, min_height: float = 0.0, max_height: float | None = None) -> gpd.GeoDataFrame:
    if len(buildings) == 0:
        return buildings
    keep = buildings["height_m"] >= min_height
    if max_height is not None:
        keep &= buildings["height_m"] <= max_height
    return buildings[keep].reset_index(drop=True)


def find_buildings(point_cloud: PointCloud, min_height: float = 20, source: str = "predicted") -> gpd.GeoDataFrame:
    """Roadmap phase-3 deliverable: all buildings taller than `min_height` metres."""
    return filter_by_height(segment_buildings(point_cloud, source=source), min_height=min_height)
