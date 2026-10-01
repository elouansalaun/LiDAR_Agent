"""Ground normalisation, features and building analysis on synthetic scenes."""

import numpy as np
import pytest

from src import config
from src.buildings.analysis import find_buildings, segment_buildings
from src.lidar import features, ground, io
from src.lidar.point_cloud import PointCloud
from tests.synthetic import make_scene


def test_height_above_ground_on_sloping_terrain():
    cloud = make_scene()
    cloud, _, method = ground.normalize_height(cloud)

    assert method == "classified"
    hag = cloud["height_above_ground"]
    ground_points = cloud["classification"] == config.ASPRS_GROUND
    assert np.abs(hag[ground_points]).max() < 0.2
    # Roofs are 15 m and 30 m above a slope that rises 5 m across the scene.
    roofs = hag[~ground_points]
    assert set(np.round(np.unique(np.round(roofs))).astype(int)) <= {14, 15, 16, 29, 30, 31}


def test_ground_estimate_without_classification():
    cloud = make_scene(buildings=[(40, 40, 20, 12.0)])
    cloud.data["classification"][:] = config.ASPRS_UNCLASSIFIED  # pretend the file is unclassified
    cloud, _, method = ground.normalize_height(cloud)

    assert method == "estimate"
    roof = cloud["z"] > 150 + 0.05 * 100 + 5
    assert np.median(cloud["height_above_ground"][roof]) == pytest.approx(12.0, abs=0.5)
    assert np.median(np.abs(cloud["height_above_ground"][~roof])) < 0.2


def test_geometric_features_distinguish_flat_and_scattered_points():
    rng = np.random.default_rng(1)
    flat = np.column_stack([rng.uniform(0, 10, 3000), rng.uniform(0, 10, 3000), np.zeros(3000)])
    crown = rng.normal(0, 1.5, (3000, 3)) + [20, 5, 10]
    xyz = np.vstack([flat, crown])
    cloud = PointCloud({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                        "height_above_ground": xyz[:, 2].astype(np.float32)})
    features.compute_features(cloud)

    is_flat = np.arange(len(cloud)) < 3000
    assert np.median(cloud["scattering"][is_flat]) < 0.01      # no thickness at all
    assert np.median(cloud["verticality"][is_flat]) < 0.05     # horizontal surface
    assert np.median(cloud["scattering"][~is_flat]) > 0.3      # volumetric blob
    assert np.median(cloud["planarity"][is_flat]) > 2 * np.median(cloud["planarity"][~is_flat])


def test_segment_buildings_finds_heights_and_areas():
    cloud = make_scene()
    ground.normalize_height(cloud)
    buildings = segment_buildings(cloud)

    assert len(buildings) == 2
    tall, short = buildings.iloc[0], buildings.iloc[1]
    assert tall.height_m == pytest.approx(30.0, abs=0.5)
    assert short.height_m == pytest.approx(15.0, abs=0.5)
    assert tall.area_m2 == pytest.approx(400, rel=0.1)
    assert buildings.crs.to_epsg() == 2950
    assert buildings.geometry.is_valid.all()


def test_find_buildings_filters_by_height():
    cloud = make_scene()
    ground.normalize_height(cloud)
    assert len(find_buildings(cloud, min_height=20)) == 1
    assert len(find_buildings(cloud, min_height=40)) == 0


def test_save_and_read_roundtrip(tmp_path):
    cloud = make_scene(size_m=20, buildings=[])
    ground.normalize_height(cloud)
    path = io.save_point_cloud(cloud, tmp_path / "scene.laz", extra_fields=["height_above_ground"])
    loaded = io.read_point_cloud(path)

    assert len(loaded) == len(cloud)
    assert "height_above_ground" in loaded and "predicted_class" not in loaded
    assert loaded.crs.to_epsg() == 2950
    np.testing.assert_allclose(loaded["z"], cloud["z"], atol=0.006)  # 1 cm storage precision
