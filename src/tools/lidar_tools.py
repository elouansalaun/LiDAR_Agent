"""Agent tools for reading and preprocessing LiDAR data."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

from src import config
from src.lidar import ground, io
from src.lidar.pipeline import area_name
from src.tools.files import derived_path, is_up_to_date, relative, resolve_input
from src.tools.registry import ToolError, tool


@tool
def find_lidar_tiles() -> dict:
    """List the available LiDAR tiles with their extent (in EPSG:2950 and in lon/lat), point count and density.
    Call this first to know which area is covered before loading data."""
    tiles = io.find_tiles()
    if not tiles:
        raise ToolError(f"No .las/.laz files found in {relative(config.RAW_TILES_DIR)}")

    coverage = (
        min(t.bounds[0] for t in tiles), min(t.bounds[1] for t in tiles),
        max(t.bounds[2] for t in tiles), max(t.bounds[3] for t in tiles),
    )
    return {
        "crs": tiles[0].crs,
        "coverage_bbox": coverage,
        "coverage_bbox_lonlat": [round(v, 6) for v in io.transform_bbox(coverage, tiles[0].crs, "EPSG:4326")],
        "max_area_per_request_km2": config.MAX_AREA_KM2,
        "tiles": [
            {
                "name": t.name,
                "bbox": t.bounds,
                "bbox_lonlat": [round(v, 6) for v in t.bounds_lonlat],
                "point_count": t.point_count,
                "density_pts_per_m2": round(t.point_count / (t.area_km2 * 1e6), 1),
            }
            for t in tiles
        ],
    }


@tool
def load_point_cloud(
    tile_name: Annotated[str | None, Field(description="Load a whole tile (1 km², slowest option)")] = None,
    bbox: Annotated[list[float] | None, Field(description="[min_x, min_y, max_x, max_y] of the area to crop", min_length=4, max_length=4)] = None,
    bbox_crs: Annotated[Literal["EPSG:2950", "EPSG:4326"], Field(description="CRS of bbox: EPSG:2950 metres (tile CRS) or EPSG:4326 lon/lat")] = "EPSG:2950",
    center_lon: Annotated[float | None, Field(description="Longitude of the centre of a square area (use with center_lat)")] = None,
    center_lat: Annotated[float | None, Field(description="Latitude of the centre of a square area (use with center_lon)")] = None,
    size_m: Annotated[float, Field(description="Side of the square area around the centre, in metres", gt=0, le=1000)] = 500,
) -> dict:
    """Crop the LiDAR tiles to an area and save it as a .laz file (noise points removed).
    Give exactly one of: tile_name, bbox, or center_lon + center_lat (+ size_m).
    Returns the path of the cropped point cloud, to pass to normalize_height."""
    tiles = io.find_tiles()
    crs = tiles[0].crs

    given = [tile_name is not None, bbox is not None, center_lon is not None or center_lat is not None]
    if sum(given) != 1:
        raise ToolError("Give exactly one of: tile_name, bbox, or center_lon + center_lat.")

    if tile_name is not None:
        area = io.get_tile(tile_name).bounds
    elif bbox is not None:
        area = tuple(bbox) if bbox_crs == "EPSG:2950" else io.transform_bbox(tuple(bbox), "EPSG:4326", crs)
    else:
        if center_lon is None or center_lat is None:
            raise ToolError("center_lon and center_lat must be given together.")
        x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(center_lon, center_lat)
        area = (x - size_m / 2, y - size_m / 2, x + size_m / 2, y + size_m / 2)

    area = tuple(round(v) for v in area)
    area_km2 = (area[2] - area[0]) * (area[3] - area[1]) / 1e6
    if area[2] <= area[0] or area[3] <= area[1]:
        raise ToolError(f"Invalid bbox {area}: expected [min_x, min_y, max_x, max_y].")
    if area_km2 > config.MAX_AREA_KM2 + 1e-6:
        raise ToolError(f"Area is {area_km2:.2f} km², the limit is {config.MAX_AREA_KM2} km². Use a smaller area.")

    output = config.PROCESSED_DIR / f"{area_name(area)}.laz"
    reused = output.exists()  # raw tiles never change, so an existing crop is always valid
    cloud = io.read_point_cloud(output) if reused else io.load_area(area, tiles)
    if not reused:
        io.save_point_cloud(cloud, output, extra_fields=[])

    coverage = unary_union([box(*t.bounds) for t in tiles])
    covered_fraction = box(*area).intersection(coverage).area / box(*area).area

    result = {
        "point_cloud_path": relative(output),
        "point_count": len(cloud),
        "bbox": area,
        "bbox_lonlat": [round(v, 6) for v in io.transform_bbox(area, crs, "EPSG:4326")],
        "area_km2": round(area_km2, 3),
        "crs": crs,
        "source_tiles": [t.name for t in tiles if io.bboxes_intersect(t.bounds, area)],
        "reused_existing_file": reused,
    }
    if covered_fraction < 0.99:
        result["note"] = f"Only {covered_fraction:.0%} of this area is inside the LiDAR coverage; the rest has no data."
    return result


@tool
def normalize_height(
    point_cloud_path: Annotated[str, Field(description="Path returned by load_point_cloud")],
    method: Annotated[Literal["auto", "classified", "estimate"], Field(
        description="Ground source: 'classified' uses the ground class shipped with the data, "
                    "'estimate' detects the ground itself, 'auto' picks classified when available")] = "auto",
) -> dict:
    """Compute each point's height above ground (Z minus a 1 m terrain model).
    Required before classification and building extraction: raw Z values are altitudes above sea level, not heights."""
    source = resolve_input(point_cloud_path)
    output = derived_path(source, "hag")
    dtm_path = derived_path(source, "dtm", ".tif")
    summary_path = output.with_suffix(".json")

    # Rewriting the file would make the (slow) classification look out of date, so reuse it.
    if is_up_to_date(summary_path, source):
        previous = json.loads(summary_path.read_text())
        if previous["requested_method"] == method:
            return {**previous, "reused_existing_file": True}

    cloud = io.read_point_cloud(source)
    if len(cloud) == 0:
        raise ToolError("The point cloud is empty.")
    cloud, dtm, used_method = ground.normalize_height(cloud, method=method)
    io.save_point_cloud(cloud, output, extra_fields=["height_above_ground"])
    dtm.save(dtm_path)

    hag = cloud["height_above_ground"]
    result = {
        "point_cloud_path": relative(output),
        "dtm_path": relative(dtm_path),
        "requested_method": method,
        "ground_method": used_method,
        "point_count": len(cloud),
        "crs": cloud.crs.to_string() if cloud.crs else None,
        "height_above_ground_m": {
            "min": round(float(hag.min()), 2),
            "p1": round(float(np.percentile(hag, 1)), 2),
            "median": round(float(np.median(hag)), 2),
            "p99": round(float(np.percentile(hag, 99)), 2),
            "max": round(float(hag.max()), 2),
        },
    }
    summary_path.write_text(json.dumps(result))
    return {**result, "reused_existing_file": False}
