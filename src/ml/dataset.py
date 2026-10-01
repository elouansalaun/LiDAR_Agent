"""Build a training / test set of labelled points from the raw tiles.

We cannot feed 130 million points to a Random Forest, and we do not need to.
Each tile is cut into a grid of non-overlapping 200 m x 200 m blocks. A few
blocks per tile are picked at random for training and a few *other* blocks
for testing. For each block we run the phase-1 pipeline (features need the
full-density neighbourhood), then keep a random sample of its points.

Why blocks? The split has to be *spatial*. Splitting random points would
leak information, because neighbouring points share almost identical
features. Holding out a whole tile was tried first, but the held-out tile was
mostly Mount Royal park (0.8% building points), which made the building
metrics meaningless. Held-out blocks spread over all tiles give a test set
that looks like the rest of the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.lidar import features, ground, io
from src.ml.labels import asprs_to_target


@dataclass
class Samples:
    X: np.ndarray             # (N, F) features
    y: np.ndarray             # (N,) target classes
    feature_names: list[str]
    block_ids: np.ndarray     # which block each point came from (for inspection)


def block_grid(tile: io.TileInfo, size_m: float) -> list[io.BBox]:
    """All non-overlapping square blocks of `size_m` inside a tile."""
    min_x, min_y, max_x, max_y = tile.bounds
    xs = np.arange(min_x, max_x - size_m + 1, size_m)
    ys = np.arange(min_y, max_y - size_m + 1, size_m)
    return [(float(x), float(y), float(x + size_m), float(y + size_m)) for x in xs for y in ys]


def split_blocks(
    n_train_per_tile: int = 5,
    n_test_per_tile: int = 2,
    size_m: float = 200.0,
    seed: int = 42,
) -> tuple[list[io.BBox], list[io.BBox]]:
    """Randomly choose disjoint train and test blocks in every tile."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for tile in io.find_tiles():
        blocks = block_grid(tile, size_m)
        chosen = rng.choice(len(blocks), size=n_train_per_tile + n_test_per_tile, replace=False)
        train += [blocks[i] for i in chosen[:n_train_per_tile]]
        test += [blocks[i] for i in chosen[n_train_per_tile:]]
    return train, test


def sample_block(bbox: io.BBox, points_per_block: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Run the pipeline on one block and return a random sample of (features, labels)."""
    cloud = io.load_area(bbox)
    cloud, _, _ = ground.normalize_height(cloud, method="classified")
    cloud = features.compute_features(cloud, geometric=True)

    X = features.feature_matrix(cloud, features.ALL_FEATURES)
    y = asprs_to_target(cloud["classification"], cloud["height_above_ground"])

    keep = rng.choice(len(y), size=min(points_per_block, len(y)), replace=False)
    return X[keep], y[keep]


def build_samples(blocks: list[io.BBox], points_per_block: int, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    X_parts, y_parts, block_parts = [], [], []
    for block_id, bbox in enumerate(blocks):
        print(f"  block {block_id + 1}/{len(blocks)} {bbox} ...", flush=True)
        X, y = sample_block(bbox, points_per_block, rng)
        X_parts.append(X)
        y_parts.append(y)
        block_parts.append(np.full(len(y), block_id))

    return Samples(
        X=np.concatenate(X_parts),
        y=np.concatenate(y_parts),
        feature_names=list(features.ALL_FEATURES),
        block_ids=np.concatenate(block_parts),
    )


def load_or_build(path: Path, blocks: list[io.BBox], points_per_block: int) -> Samples:
    """Cache samples to disk: building them takes a few minutes."""
    if path.exists():
        data = np.load(path)
        return Samples(data["X"], data["y"], list(data["feature_names"]), data["block_ids"])

    samples = build_samples(blocks, points_per_block)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=samples.X, y=samples.y,
                        feature_names=np.array(samples.feature_names), block_ids=samples.block_ids)
    return samples


def train_and_test_samples() -> tuple[Samples, Samples]:
    train_blocks, test_blocks = split_blocks()
    print(f"{len(train_blocks)} training blocks, {len(test_blocks)} test blocks (200 m x 200 m)")
    train = load_or_build(config.PROCESSED_DIR / "train_samples.npz", train_blocks, points_per_block=40_000)
    test = load_or_build(config.PROCESSED_DIR / "test_samples.npz", test_blocks, points_per_block=60_000)
    return train, test
