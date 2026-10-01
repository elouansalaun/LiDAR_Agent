"""How good are the buildings we extract? Compare against the vendor classification.

For every held-out test block of the classifier (never seen in training), we
extract buildings twice with the same segmentation code:
    * from our Random Forest predictions
    * from the vendor's ASPRS building class (the reference)
and compare footprints and heights.

    python -m src.buildings.evaluate
"""

from __future__ import annotations

import json

import numpy as np
from shapely.ops import unary_union

from src import config
from src.buildings.analysis import segment_buildings
from src.lidar import ground, io
from src.ml.classify import classify_lidar_points
from src.ml.dataset import split_blocks

MIN_MATCH_AREA_M2 = 100  # ignore sheds and fragments when matching buildings


def compare_block(bbox: io.BBox) -> dict:
    cloud = io.load_area(bbox)
    ground.normalize_height(cloud, method="classified")
    classify_lidar_points(cloud)
    predicted = segment_buildings(cloud, source="predicted")
    reference = segment_buildings(cloud, source="vendor")

    # Footprint agreement over the whole block.
    union_p, union_r = unary_union(predicted.geometry), unary_union(reference.geometry)
    intersection = union_p.intersection(union_r).area
    union = union_p.union(union_r).area

    # One-to-one matches: most overlapping predicted building, similar footprint.
    height_errors, tall_found, tall_total = [], 0, 0
    for ref in reference[reference.area_m2 >= MIN_MATCH_AREA_M2].itertuples():
        candidates = predicted[predicted.intersects(ref.geometry)]
        overlap = candidates.geometry.intersection(ref.geometry).area if len(candidates) else None
        if ref.height_m >= 20:
            tall_total += 1
            tall_found += bool(len(candidates)) and bool((overlap > 0.5 * ref.area_m2).any())
        if not len(candidates):
            continue
        best = candidates.iloc[int(np.argmax(overlap.values))]
        if overlap.max() > 0.5 * ref.area_m2 and abs(best.area_m2 - ref.area_m2) < 0.3 * ref.area_m2:
            height_errors.append(best.height_m - ref.height_m)

    return {
        "bbox": bbox,
        "predicted_buildings": len(predicted),
        "reference_buildings": len(reference),
        "reference_buildings_ge_100m2": int((reference.area_m2 >= MIN_MATCH_AREA_M2).sum()),
        "matched_one_to_one": len(height_errors),
        "height_errors_m": [round(e, 2) for e in height_errors],
        "footprint_intersection_m2": intersection,
        "footprint_union_m2": union,
        "tall_reference": tall_total,
        "tall_found": int(tall_found),
    }


def main() -> None:
    _, test_blocks = split_blocks()
    results = []
    for bbox in test_blocks:
        result = compare_block(bbox)
        results.append(result)
        print(f"{bbox}: predicted {result['predicted_buildings']:3d}  reference {result['reference_buildings']:3d}  "
              f"matched {result['matched_one_to_one']:3d}/{result['reference_buildings_ge_100m2']}")

    errors = np.abs(np.concatenate([r["height_errors_m"] for r in results]))
    summary = {
        "test_blocks": len(results),
        "reference_buildings_ge_100m2": sum(r["reference_buildings_ge_100m2"] for r in results),
        "matched_one_to_one": sum(r["matched_one_to_one"] for r in results),
        "height_abs_error_median_m": round(float(np.median(errors)), 2),
        "height_abs_error_p90_m": round(float(np.percentile(errors, 90)), 2),
        "footprint_iou": round(sum(r["footprint_intersection_m2"] for r in results) / sum(r["footprint_union_m2"] for r in results), 3),
        "buildings_ge_20m_found": f"{sum(r['tall_found'] for r in results)}/{sum(r['tall_reference'] for r in results)}",
    }
    print(json.dumps(summary, indent=2))
    (config.MODELS_DIR / "building_evaluation.json").write_text(json.dumps({"summary": summary, "blocks": results}, indent=2, default=float))


if __name__ == "__main__":
    main()
