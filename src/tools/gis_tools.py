"""Agent tools for building analysis and maps."""

from __future__ import annotations

from typing import Annotated

import geopandas as gpd
from pydantic import Field

from src import config
from src.buildings import analysis
from src.lidar import io
from src.tools.files import derived_path, relative, resolve_input
from src.tools.registry import ToolError, tool
from src.visualization import maps

MAX_BUILDINGS_LISTED = 25  # keep tool results short for the LLM; the full list is in the file


def summarize_buildings(buildings: gpd.GeoDataFrame) -> dict:
    """Short, LLM-friendly description of a set of buildings (tallest first)."""
    if len(buildings) == 0:
        return {"building_count": 0, "buildings": []}
    heights = buildings["height_m"]
    listed = buildings.sort_values("height_m", ascending=False).head(MAX_BUILDINGS_LISTED)
    return {
        "building_count": len(buildings),
        "height_m": {"min": round(float(heights.min()), 1), "mean": round(float(heights.mean()), 1),
                     "max": round(float(heights.max()), 1)},
        "total_footprint_m2": round(float(buildings["area_m2"].sum()), 0),
        "buildings": [
            {"id": row.building_id, "height_m": row.height_m, "max_height_m": row.max_height_m,
             "area_m2": row.area_m2, "lat": row.lat, "lon": row.lon}
            for row in listed.itertuples()
        ],
        "buildings_listed": min(len(buildings), MAX_BUILDINGS_LISTED),
    }


@tool
def extract_buildings(
    point_cloud_path: Annotated[str, Field(description="Path returned by classify_lidar_points")],
    min_area_m2: Annotated[float, Field(description="Ignore footprints smaller than this (m²)", ge=1)] = config.BUILDING_MIN_AREA_M2,
    height_percentile: Annotated[float, Field(description="Percentile of point heights used as the building height (98 = robust to antennas)", ge=50, le=100)] = config.BUILDING_HEIGHT_PERCENTILE,
) -> dict:
    """Group building points into individual buildings and measure each one: height, max height, footprint area and centroid.
    Saves the footprints as GeoJSON (lon/lat) and returns summary statistics, tallest buildings first."""
    source = resolve_input(point_cloud_path)
    cloud = io.read_point_cloud(source)
    if "predicted_class" not in cloud:
        raise ToolError("This point cloud is not classified. Call classify_lidar_points on it first.")

    buildings = analysis.segment_buildings(cloud, min_area_m2=min_area_m2, height_percentile=height_percentile)
    output = derived_path(source, "buildings", ".geojson")
    _save_geojson(buildings, output)

    return {
        "buildings_path": relative(output),
        "crs_of_source": cloud.crs.to_string() if cloud.crs else None,
        "area_bbox_lonlat": _bbox_lonlat(cloud),
        "invalid_geometries": int((~buildings.geometry.is_valid).sum()) if len(buildings) else 0,
        **summarize_buildings(buildings),
    }


@tool
def filter_buildings(
    buildings_path: Annotated[str, Field(description="GeoJSON returned by extract_buildings")],
    min_height: Annotated[float, Field(description="Keep buildings at least this tall (m)", ge=0)] = 0,
    max_height: Annotated[float | None, Field(description="Keep buildings at most this tall (m); omit for no limit", ge=0)] = None,
) -> dict:
    """Keep only the buildings whose height is within [min_height, max_height] metres.
    Saves the selection as a new GeoJSON (to pass to create_map) and lists the matching buildings."""
    source = resolve_input(buildings_path)
    buildings = gpd.read_file(source)
    selected = analysis.filter_by_height(buildings, min_height=min_height, max_height=max_height)

    suffix = f"min{min_height:g}m" + (f"_max{max_height:g}m" if max_height is not None else "")
    output = derived_path(source, suffix)
    _save_geojson(selected, output)
    return {"buildings_path": relative(output), "filter": {"min_height": min_height, "max_height": max_height},
            **summarize_buildings(selected)}


@tool
def create_map(
    buildings_path: Annotated[str, Field(description="GeoJSON returned by extract_buildings or filter_buildings")],
    title: Annotated[str, Field(description="Layer title shown on the map")] = "Buildings",
    area_bbox_lonlat: Annotated[list[float] | None, Field(description="Optional [min_lon, min_lat, max_lon, max_lat] of the analysed area, drawn as an outline", min_length=4, max_length=4)] = None,
) -> dict:
    """Create an interactive HTML map of building footprints coloured by height (with a satellite layer and tooltips).
    Returns the path of the HTML file."""
    source = resolve_input(buildings_path)
    buildings = gpd.read_file(source)
    if len(buildings) == 0 and area_bbox_lonlat is None:
        raise ToolError("No buildings to map. Pass area_bbox_lonlat to map the (empty) area anyway.")

    fmap = maps.building_map(buildings, title=title, area_bbox_lonlat=area_bbox_lonlat)
    output = maps.save_map(fmap, config.OUTPUTS_DIR / f"{source.stem}_map.html")
    return {"map_path": relative(output), "building_count": len(buildings)}


def _save_geojson(buildings: gpd.GeoDataFrame, path) -> None:
    # GeoJSON is lon/lat by specification; the metric coordinates stay available in centroid_x / centroid_y.
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(buildings) == 0:
        path.write_text('{"type": "FeatureCollection", "features": []}')
        return
    buildings.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")


def _bbox_lonlat(cloud) -> list[float] | None:
    if cloud.crs is None:
        return None
    return [round(v, 6) for v in io.transform_bbox(cloud.bounds, cloud.crs, "EPSG:4326")]
