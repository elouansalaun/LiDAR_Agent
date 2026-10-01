"""A minimal in-memory point cloud: a dict of NumPy arrays plus a CRS.

laspy's LasData object is great for reading/writing files, but awkward to
merge, filter and extend with new attributes. Inside the pipeline we use this
small class instead, and convert back to LAS only when saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pyproj import CRS


@dataclass
class PointCloud:
    """Column-oriented point cloud.

    `data` maps an attribute name ("x", "y", "z", "intensity", ...) to a 1-D
    array. All arrays have the same length (one value per point).
    """

    data: dict[str, np.ndarray]
    crs: CRS | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data["x"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name]

    def __contains__(self, name: str) -> bool:
        return name in self.data

    @property
    def xyz(self) -> np.ndarray:
        """(N, 3) array of coordinates."""
        return np.column_stack([self.data["x"], self.data["y"], self.data["z"]])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)."""
        x, y = self.data["x"], self.data["y"]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def add(self, name: str, values: np.ndarray) -> None:
        """Add (or overwrite) a per-point attribute."""
        values = np.asarray(values)
        if len(values) != len(self):
            raise ValueError(f"'{name}' has {len(values)} values, expected {len(self)}")
        self.data[name] = values

    def subset(self, mask: np.ndarray) -> PointCloud:
        """Return a new point cloud with only the points where `mask` is True (or the given indices)."""
        return PointCloud(
            data={name: values[mask] for name, values in self.data.items()},
            crs=self.crs,
            metadata=dict(self.metadata),
        )

    @staticmethod
    def concatenate(clouds: list[PointCloud]) -> PointCloud:
        """Merge several clouds that share the same attributes and CRS."""
        if not clouds:
            raise ValueError("Nothing to concatenate")
        names = clouds[0].data.keys()
        return PointCloud(
            data={name: np.concatenate([c.data[name] for c in clouds]) for name in names},
            crs=clouds[0].crs,
            metadata=dict(clouds[0].metadata),
        )
