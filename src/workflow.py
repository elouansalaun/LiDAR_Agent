"""The building workflow without an LLM: the same tools, in a fixed order.

Useful to demo the project without an API key, to benchmark the pipeline,
and as a reference for what the agent is expected to do.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

from src.agent.validator import validate_results
from src.tools import TOOLS, ToolError


def run_building_query(area: dict, min_height: float = 20.0, on_step: Callable[[dict], None] | None = None) -> dict:
    """Run load -> normalize -> classify -> extract -> filter -> map on `area`.

    `area` holds the arguments of load_point_cloud (tile_name, bbox or center_lon/center_lat/size_m).
    Returns every step's result plus validation issues. Raises ToolError if a step fails.
    """
    steps = []

    def run(name: str, **arguments) -> dict:
        start = time.perf_counter()
        result = TOOLS[name].run(arguments)
        issues = validate_results([(name, json.dumps(result), False)]).issues
        step = {"name": name, "input": arguments, "output": result, "seconds": round(time.perf_counter() - start, 1),
                "issues": [f"{i.severity}: {i.message}" for i in issues]}
        steps.append(step)
        if on_step:
            on_step(step)
        if any(i.severity == "error" for i in issues):
            raise ToolError(f"{name} failed validation: {step['issues']}")
        return result

    loaded = run("load_point_cloud", **area)
    normalized = run("normalize_height", point_cloud_path=loaded["point_cloud_path"])
    classified = run("classify_lidar_points", point_cloud_path=normalized["point_cloud_path"])
    extracted = run("extract_buildings", point_cloud_path=classified["point_cloud_path"])
    selected = run("filter_buildings", buildings_path=extracted["buildings_path"], min_height=min_height)
    mapped = run("create_map", buildings_path=selected["buildings_path"], title=f"Buildings ≥ {min_height:g} m",
                 area_bbox_lonlat=extracted["area_bbox_lonlat"])

    return {
        "area": loaded,
        "class_percentages": classified["class_percentages"],
        "classified_point_cloud_path": classified["point_cloud_path"],
        "all_buildings": extracted,
        "selected_buildings": selected,
        "map_path": mapped["map_path"],
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lon", type=float, default=-73.6126, help="centre longitude")
    parser.add_argument("--lat", type=float, default=45.4999, help="centre latitude")
    parser.add_argument("--size", type=float, default=500, help="side of the square area (m)")
    parser.add_argument("--min-height", type=float, default=20)
    args = parser.parse_args()

    result = run_building_query(
        {"center_lon": args.lon, "center_lat": args.lat, "size_m": args.size},
        min_height=args.min_height,
        on_step=lambda s: print(f"  {s['name']:<22} {s['seconds']:>5}s  {'; '.join(s['issues'])}"),
    )
    selected = result["selected_buildings"]
    print(f"\n{selected['building_count']} buildings ≥ {args.min_height:g} m")
    for b in selected["buildings"]:
        print(f"  {b['id']}  {b['height_m']:6.1f} m  (highest point {b['max_height_m']:.1f} m)  {b['area_m2']:8.0f} m²  {b['lat']:.5f}, {b['lon']:.5f}")
    print(f"Map: {result['map_path']}")


if __name__ == "__main__":
    main()
