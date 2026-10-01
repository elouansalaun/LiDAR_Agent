"""Turn the vendor ASPRS classification into our 4 training classes.

The Montreal tiles only contain these ASPRS codes:
    1 unclassified, 2 ground, 6 building, 7 noise, 17 bridge deck

There is no vegetation class, so vegetation labels are *derived*:
unclassified points more than 2 m above the ground are called vegetation.
In this residential/campus area that is overwhelmingly trees, but it also
catches a few wires, poles and street lights. These are "weak" labels:
good enough to train a baseline, and the limitation is documented in
docs/03_classification.md.
"""

import numpy as np

from src import config


def asprs_to_target(classification: np.ndarray, height_above_ground: np.ndarray) -> np.ndarray:
    """Map ASPRS codes (+ height) to 0=ground, 1=vegetation, 2=building, 3=other."""
    target = np.full(len(classification), config.OTHER, dtype=np.uint8)
    target[classification == config.ASPRS_GROUND] = config.GROUND
    target[classification == config.ASPRS_BUILDING] = config.BUILDING

    tall_unclassified = (classification == config.ASPRS_UNCLASSIFIED) & (
        height_above_ground > config.VEGETATION_MIN_HEIGHT_M
    )
    target[tall_unclassified] = config.VEGETATION
    return target
