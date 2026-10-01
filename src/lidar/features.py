"""Per-point features used by the classifier.

Baseline features (from the roadmap):
    height_above_ground, intensity, return_number, number_of_returns, local_density

Geometric features (added after the first experiments, see docs/03_classification.md):
    linearity, planarity, scattering, verticality

The geometric features describe the *shape* of each point's neighbourhood,
computed from the eigenvalues of the 3x3 covariance matrix of its k nearest
neighbours (λ1 >= λ2 >= λ3):

    linearity   = (λ1 - λ2) / λ1   -> high on wires, poles, roof edges
    planarity   = (λ2 - λ3) / λ1   -> high on roofs, walls, roads
    scattering  =  λ3 / λ1         -> high in tree crowns (volumetric)
    verticality = 1 - |n_z|        -> 0 on flat roofs, 1 on walls

These make "flat roof vs. tree crown" separable even when both are 15 m high,
which height + intensity alone struggle with.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src import config
from src.lidar.point_cloud import PointCloud

BASELINE_FEATURES = [
    "height_above_ground",
    "intensity",
    "return_number",
    "number_of_returns",
    "local_density",
]
GEOMETRIC_FEATURES = ["linearity", "planarity", "scattering", "verticality"]
ALL_FEATURES = BASELINE_FEATURES + GEOMETRIC_FEATURES


def compute_local_density(tree: cKDTree, points: np.ndarray, radius: float = config.DENSITY_RADIUS_M) -> np.ndarray:
    """Number of neighbours inside a sphere of `radius` metres around each point."""
    counts = tree.query_ball_point(points, r=radius, return_length=True, workers=-1)
    return counts.astype(np.float32)


def compute_geometric_features(
    tree: cKDTree,
    points: np.ndarray,
    k: int = config.NEIGHBOURS_K,
    chunk_size: int = 500_000,
) -> dict[str, np.ndarray]:
    """Eigenvalue-based shape descriptors (see module docstring).

    Processed in chunks: for 1M points the neighbourhood array alone is
    1M x 16 x 3 floats (~400 MB), so we avoid doing the whole cloud at once.
    """
    n = len(points)
    out = {name: np.empty(n, dtype=np.float32) for name in GEOMETRIC_FEATURES}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        _, idx = tree.query(points[start:end], k=k, workers=-1)

        neighbours = points[idx]                                  # (m, k, 3)
        neighbours = neighbours - neighbours.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", neighbours, neighbours) / k
        eigenvalues, eigenvectors = np.linalg.eigh(cov)          # ascending order

        l3, l2, l1 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        l1 = np.maximum(l1, 1e-9)  # avoid division by zero on duplicated points
        normal_z = eigenvectors[:, 2, 0]  # z component of the smallest eigenvector

        out["linearity"][start:end] = (l1 - l2) / l1
        out["planarity"][start:end] = (l2 - l3) / l1
        out["scattering"][start:end] = l3 / l1
        out["verticality"][start:end] = 1.0 - np.abs(normal_z)

    return out


def compute_features(cloud: PointCloud, geometric: bool = True) -> PointCloud:
    """Add local density (and optionally geometric features) to the cloud.

    `height_above_ground` must already exist (see ground.normalize_height).
    """
    if "height_above_ground" not in cloud:
        raise ValueError("Run normalize_height() before compute_features()")

    # Work in local coordinates: keeps the covariance maths precise.
    points = cloud.xyz - cloud.xyz.min(axis=0)
    tree = cKDTree(points)

    cloud.add("local_density", compute_local_density(tree, points))
    if geometric:
        for name, values in compute_geometric_features(tree, points).items():
            cloud.add(name, values)
    return cloud


def feature_matrix(cloud: PointCloud, names: list[str]) -> np.ndarray:
    """Stack the requested attributes into an (N, F) float32 matrix for scikit-learn."""
    missing = [n for n in names if n not in cloud]
    if missing:
        raise ValueError(f"Missing features {missing}. Did you run compute_features()?")
    return np.column_stack([cloud[n].astype(np.float32) for n in names])
