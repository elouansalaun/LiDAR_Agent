"""Static figures for the documentation (PNG, matplotlib).

    python -m src.visualization.figures

Runs the workflow on the Saint Joseph's Oratory area (cached after the first
run) and writes docs/images/oratory_classes.png and oratory_buildings.png.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from src import config
from src.lidar import io
from src.lidar.point_cloud import PointCloud
from src.visualization.maps import CLASS_COLORS, HEIGHT_RAMP

DOCS_IMAGES = config.PROJECT_ROOT / "docs" / "images"
INK, MUTED = "#0b0b0b", "#898781"


def classes_top_view(cloud: PointCloud, path: Path, title: str, max_points: int = 600_000) -> Path:
    """Top-down view of the classified points (drawing order: ground first, buildings last)."""
    rng = np.random.default_rng(0)
    sub = cloud.subset(rng.choice(len(cloud), size=min(max_points, len(cloud)), replace=False))
    x, y = sub["x"] - sub["x"].min(), sub["y"] - sub["y"].min()

    fig, ax = plt.subplots(figsize=(7, 7))
    for label in (config.GROUND, config.OTHER, config.VEGETATION, config.BUILDING):
        mask = sub["predicted_class"] == label
        ax.scatter(x[mask], y[mask], s=0.2, c=CLASS_COLORS[label], linewidths=0, rasterized=True)

    handles = [Patch(color=CLASS_COLORS[k], label=config.CLASS_NAMES[k]) for k in (config.BUILDING, config.VEGETATION, config.GROUND, config.OTHER)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, labelcolor=INK)
    _style(ax, title)
    return _save(fig, path)


def buildings_by_height(buildings: gpd.GeoDataFrame, path: Path, title: str, min_height: float = 20) -> Path:
    """Footprints: grey below `min_height`, blue ramp (light -> dark = low -> high) above."""
    buildings = buildings.to_crs("EPSG:2950")
    x0, y0 = buildings.total_bounds[:2]
    buildings = buildings.set_geometry(buildings.geometry.translate(-x0, -y0))
    tall = buildings[buildings["height_m"] >= min_height]

    fig, ax = plt.subplots(figsize=(7.6, 7))
    buildings[buildings["height_m"] < min_height].plot(ax=ax, color="#e1e0d9", edgecolor="white", linewidth=0.5)
    ramp = LinearSegmentedColormap.from_list("height", HEIGHT_RAMP)
    tall.plot(ax=ax, column="height_m", cmap=ramp, edgecolor="white", linewidth=0.8,
              legend=True, legend_kwds={"label": "Building height (m)", "shrink": 0.7})
    for row in tall.head(5).itertuples():
        point = row.geometry.representative_point()
        ax.annotate(f"{row.building_id}\n{row.height_m:.0f} m", (point.x, point.y), ha="center", va="center",
                    fontsize=8, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.legend(handles=[Patch(color="#e1e0d9", label=f"< {min_height:g} m")], loc="lower left", frameon=False)
    _style(ax, title)
    return _save(fig, path)


def _style(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("x (m)", color=MUTED)
    ax.set_ylabel("y (m)", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    from src.workflow import run_building_query

    result = run_building_query({"center_lon": -73.6178, "center_lat": 45.4925, "size_m": 500}, min_height=20)
    cloud = io.read_point_cloud(config.PROJECT_ROOT / result["classified_point_cloud_path"])
    buildings = gpd.read_file(config.PROJECT_ROOT / result["all_buildings"]["buildings_path"])

    print(classes_top_view(cloud, DOCS_IMAGES / "oratory_classes.png", "Predicted classes - Saint Joseph's Oratory, 500 m x 500 m"))
    print(buildings_by_height(buildings, DOCS_IMAGES / "oratory_buildings.png", "Buildings >= 20 m (98th-percentile roof height)"))


if __name__ == "__main__":
    main()
