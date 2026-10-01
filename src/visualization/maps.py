"""Interactive maps (Folium / Leaflet) and 3D point-cloud views (Plotly)."""

from __future__ import annotations

from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go

from src import config
from src.lidar.point_cloud import PointCloud

# Single-hue sequential ramp (light -> dark blue) for building height:
# taller buildings are darker. One hue, so the order reads without a legend lookup.
HEIGHT_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# Categorical colours for the 4 classes, in a fixed order.
CLASS_COLORS = {
    config.BUILDING: "#2a78d6",    # blue
    config.VEGETATION: "#1baf7a",  # aqua-green
    config.GROUND: "#eda100",      # yellow-ochre
    config.OTHER: "#e87ba4",       # magenta
}


def building_map(buildings: gpd.GeoDataFrame, title: str = "Buildings", area_bbox_lonlat: list[float] | None = None) -> folium.Map:
    """Leaflet map with building footprints coloured by height, with tooltips."""
    buildings = buildings.to_crs("EPSG:4326")

    if area_bbox_lonlat:
        min_lon, min_lat, max_lon, max_lat = area_bbox_lonlat
    elif len(buildings):
        min_lon, min_lat, max_lon, max_lat = buildings.total_bounds
    else:
        raise ValueError("Nothing to map: no buildings and no area given")

    fmap = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=16,
                      tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(fmap)

    if area_bbox_lonlat:
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color="#52514e", weight=1.5, fill=False, dash_array="6 4", tooltip="Analysed area",
        ).add_to(fmap)

    if len(buildings):
        heights = buildings["height_m"].astype(float)
        colormap = cm.LinearColormap(HEIGHT_RAMP, vmin=float(heights.min()), vmax=max(float(heights.max()), float(heights.min()) + 1))
        colormap.caption = "Building height (m)"

        folium.GeoJson(
            buildings[["building_id", "height_m", "max_height_m", "area_m2", "geometry"]],
            name=title,
            style_function=lambda feature: {
                "fillColor": colormap(feature["properties"]["height_m"]),
                "color": "#ffffff",   # thin light outline separates adjacent footprints
                "weight": 1,
                "fillOpacity": 0.85,
            },
            highlight_function=lambda feature: {"weight": 3, "color": "#0b0b0b"},
            tooltip=folium.GeoJsonTooltip(
                fields=["building_id", "height_m", "max_height_m", "area_m2"],
                aliases=["Building", "Height (m)", "Highest point (m)", "Footprint (m²)"],
                localize=True,
            ),
        ).add_to(fmap)
        colormap.add_to(fmap)

    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def save_map(fmap: folium.Map, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(path))
    return path


def point_cloud_figure(cloud: PointCloud, max_points: int = 150_000, color_by: str = "class") -> go.Figure:
    """3D scatter of a random subset of the cloud, coloured by class or by height."""
    rng = np.random.default_rng(0)
    keep = rng.choice(len(cloud), size=min(max_points, len(cloud)), replace=False)
    sub = cloud.subset(keep)
    x = sub["x"] - sub["x"].min()
    y = sub["y"] - sub["y"].min()
    z = sub["height_above_ground"] if "height_above_ground" in sub else sub["z"] - sub["z"].min()

    marker = dict(size=1.5)
    fig = go.Figure()
    if color_by == "class" and "predicted_class" in sub:
        for label in (config.BUILDING, config.VEGETATION, config.GROUND, config.OTHER):
            mask = sub["predicted_class"] == label
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask], mode="markers", name=config.CLASS_NAMES[label],
                marker={**marker, "color": CLASS_COLORS[label]},
                hovertemplate=f"{config.CLASS_NAMES[label]}<br>height %{{z:.1f}} m<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="points",
            marker={**marker, "color": z, "colorscale": [[i / 6, c] for i, c in enumerate(HEIGHT_RAMP)],
                    "colorbar": {"title": "Height (m)"}},
            hovertemplate="height %{z:.1f} m<extra></extra>",
        ))

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="height (m)"),
        legend=dict(itemsizing="constant", orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    return fig
