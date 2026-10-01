"""Reading, cropping and saving LAS/LAZ files.

Main entry points:
    find_tiles()          -> list the raw tiles and their metadata (header only, fast)
    load_area(bbox)       -> read the points inside a bounding box, across tiles
    save_point_cloud()    -> write a PointCloud (with extra attributes) to .laz
    read_point_cloud()    -> read a file written by save_point_cloud()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS, Transformer

from src import config
from src.lidar.point_cloud import PointCloud

# Attributes we keep from the raw files. RGB, GPS time, scan angle... are not
# used by the pipeline, so we drop them to save memory and disk.
STANDARD_FIELDS = ["intensity", "return_number", "number_of_returns", "classification"]

# Bounding box convention used everywhere: (min_x, min_y, max_x, max_y)
BBox = tuple[float, float, float, float]


@dataclass
class TileInfo:
    name: str
    path: str
    point_count: int
    bounds: BBox
    z_range: tuple[float, float]
    crs: str | None
    las_version: str
    point_format: int
    bounds_lonlat: BBox | None = None

    @property
    def area_km2(self) -> float:
        min_x, min_y, max_x, max_y = self.bounds
        return (max_x - min_x) * (max_y - min_y) / 1e6

    def to_dict(self) -> dict:
        info = asdict(self)
        info["area_km2"] = round(self.area_km2, 3)
        info["density_pts_per_m2"] = round(self.point_count / (self.area_km2 * 1e6), 1)
        return info


def read_metadata(path: str | Path) -> TileInfo:
    """Read the header of a LAS/LAZ file (does not load the points)."""
    path = Path(path)
    with laspy.open(path) as reader:
        header = reader.header
        crs = header.parse_crs()

    bounds = (float(header.mins[0]), float(header.mins[1]), float(header.maxs[0]), float(header.maxs[1]))
    info = TileInfo(
        name=path.stem,
        path=str(path),
        point_count=int(header.point_count),
        bounds=bounds,
        z_range=(float(header.mins[2]), float(header.maxs[2])),
        crs=crs.to_string() if crs else None,
        las_version=str(header.version),
        point_format=header.point_format.id,
    )
    if crs is not None:
        info.bounds_lonlat = transform_bbox(bounds, crs, "EPSG:4326")
    return info


def find_tiles(directory: str | Path = config.RAW_TILES_DIR) -> list[TileInfo]:
    """List every .las/.laz file of a folder with its header metadata."""
    directory = Path(directory)
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".las", ".laz"))
    return [read_metadata(p) for p in paths]


def get_tile(name: str, directory: str | Path = config.RAW_TILES_DIR) -> TileInfo:
    """Find a tile by name (with or without extension)."""
    wanted = Path(name).stem.lower()
    for tile in find_tiles(directory):
        if tile.name.lower() == wanted:
            return tile
    raise FileNotFoundError(f"No tile named '{name}' in {directory}")


def transform_bbox(bbox: BBox, src_crs, dst_crs) -> BBox:
    """Reproject a bounding box (corner points only; fine for small areas)."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    min_x, min_y, max_x, max_y = bbox
    xs, ys = transformer.transform([min_x, max_x, min_x, max_x], [min_y, min_y, max_y, max_y])
    return (min(xs), min(ys), max(xs), max(ys))


def bboxes_intersect(a: BBox, b: BBox) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def load_area(
    bbox: BBox,
    tiles: list[TileInfo] | None = None,
    drop_noise: bool = True,
    chunk_size: int = 5_000_000,
) -> PointCloud:
    """Load every point inside `bbox` (tile CRS), reading all tiles that overlap it.

    Files are streamed in chunks so we never hold a full tile in memory when
    only a small crop is needed.
    """
    tiles = tiles if tiles is not None else find_tiles()
    overlapping = [t for t in tiles if bboxes_intersect(t.bounds, bbox)]
    if not overlapping:
        raise ValueError(f"The bounding box {bbox} does not overlap any tile")

    min_x, min_y, max_x, max_y = bbox
    pieces = []
    for tile in overlapping:
        with laspy.open(tile.path) as reader:
            crs = reader.header.parse_crs()
            for chunk in reader.chunk_iterator(chunk_size):
                x, y = np.asarray(chunk.x), np.asarray(chunk.y)
                inside = (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
                if drop_noise:
                    inside &= np.asarray(chunk.classification) != config.ASPRS_NOISE
                    inside &= ~np.asarray(chunk.withheld, dtype=bool)
                if inside.any():
                    pieces.append(_chunk_to_point_cloud(chunk, inside, crs))

    if not pieces:
        raise ValueError(f"No points found inside {bbox}")

    cloud = PointCloud.concatenate(pieces)
    cloud.metadata = {"bbox": list(bbox), "source_tiles": [t.name for t in overlapping]}
    return cloud


def load_tile(name: str) -> PointCloud:
    """Load a full tile (about 30 million points for the Montreal tiles)."""
    tile = get_tile(name)
    return load_area(tile.bounds, tiles=[tile])


def _chunk_to_point_cloud(chunk, mask: np.ndarray, crs) -> PointCloud:
    data = {
        "x": np.asarray(chunk.x)[mask],
        "y": np.asarray(chunk.y)[mask],
        "z": np.asarray(chunk.z)[mask],
    }
    for name in STANDARD_FIELDS:
        data[name] = np.asarray(chunk[name])[mask]
    return PointCloud(data=data, crs=crs)


# --- Saving / reading processed clouds ---------------------------------------


def save_point_cloud(cloud: PointCloud, path: str | Path, extra_fields: list[str] | None = None) -> Path:
    """Write a PointCloud to .laz. Non-standard attributes become LAS "extra bytes".

    extra_fields: which non-standard attributes to save (default: all of them).
    Float features compress badly, so intermediate files only keep what the
    next step needs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.floor([cloud["x"].min(), cloud["y"].min(), cloud["z"].min()])
    if cloud.crs is not None:
        header.add_crs(CRS.from_user_input(cloud.crs))

    extra_names = [n for n in cloud.data if n not in ("x", "y", "z", *STANDARD_FIELDS)]
    if extra_fields is not None:
        extra_names = [n for n in extra_names if n in extra_fields]
    for name in extra_names:
        dtype = np.uint8 if cloud[name].dtype == np.uint8 else np.float32
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))

    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud["x"], cloud["y"], cloud["z"]
    for name in STANDARD_FIELDS:
        if name in cloud:
            las[name] = cloud[name]
    for name in extra_names:
        las[name] = cloud[name].astype(np.uint8 if cloud[name].dtype == np.uint8 else np.float32)

    las.write(path)
    return path


def read_point_cloud(path: str | Path) -> PointCloud:
    """Read a .las/.laz file (raw or processed) into a PointCloud."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    las = laspy.read(path)
    data = {"x": np.asarray(las.x), "y": np.asarray(las.y), "z": np.asarray(las.z)}
    for name in STANDARD_FIELDS:
        data[name] = np.asarray(las[name])
    for name in las.point_format.extra_dimension_names:
        data[name] = np.asarray(las[name])
    return PointCloud(data=data, crs=las.header.parse_crs(), metadata={"source_file": str(path)})
