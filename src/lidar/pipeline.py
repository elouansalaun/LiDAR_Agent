"""Deliverable: the deterministic preprocessing pipeline.

    LiDAR tiles -> crop -> height normalisation -> feature extraction -> .laz

Usage:
    python -m src.lidar.pipeline --bbox 295500 5039500 296000 5040000
    python -m src.lidar.pipeline --tile 23_2955039F08_DC
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src import config
from src.lidar import features, ground, io
from src.lidar.point_cloud import PointCloud


def area_name(bbox: io.BBox) -> str:
    """Readable, deterministic name for an area, e.g. area_295500_5039500_296000_5040000."""
    return "area_" + "_".join(str(int(round(v))) for v in bbox)


def preprocess(bbox: io.BBox, output_dir: Path = config.PROCESSED_DIR, geometric: bool = True) -> tuple[PointCloud, Path]:
    """Run the whole phase-1 pipeline on an area and save the result."""
    timings = {}

    start = time.perf_counter()
    cloud = io.load_area(bbox)
    timings["load"] = time.perf_counter() - start

    start = time.perf_counter()
    cloud, dtm, method = ground.normalize_height(cloud)
    timings["normalize_height"] = time.perf_counter() - start

    start = time.perf_counter()
    cloud = features.compute_features(cloud, geometric=geometric)
    timings["features"] = time.perf_counter() - start

    name = area_name(bbox)
    output_path = io.save_point_cloud(cloud, output_dir / f"{name}_features.laz")
    dtm.save(output_dir / f"{name}_dtm.tif")

    cloud.metadata.update({"ground_method": method, "timings_s": {k: round(v, 2) for k, v in timings.items()}})
    return cloud, output_path


def summarize(cloud: PointCloud) -> str:
    hag = cloud["height_above_ground"]
    lines = [
        f"points           : {len(cloud):,}",
        f"bounds           : {tuple(round(v, 1) for v in cloud.bounds)}",
        f"CRS              : {cloud.crs.to_string() if cloud.crs else 'undefined'}",
        f"height above gnd : p1={np.percentile(hag, 1):.2f}  median={np.median(hag):.2f}  p99={np.percentile(hag, 99):.2f} m",
        f"local density    : median={np.median(cloud['local_density']):.0f} neighbours / 1 m sphere",
    ]
    for key, value in cloud.metadata.items():
        lines.append(f"{key:<17}: {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                      help="area in the tile CRS (EPSG:2950)")
    area.add_argument("--tile", help="process a whole tile (slow: ~30M points)")
    parser.add_argument("--no-geometric", action="store_true", help="only compute the baseline features")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else io.get_tile(args.tile).bounds
    cloud, path = preprocess(bbox, geometric=not args.no_geometric)
    print(summarize(cloud))
    print(f"saved to         : {path}")


if __name__ == "__main__":
    main()
