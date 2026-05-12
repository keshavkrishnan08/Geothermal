"""Shared grid utilities — define the western-US 4km grid used by every modality.

We work in EPSG:4326 for storage (lon, lat) but reproject to EPSG:5070
(Albers Equal Area Conic CONUS) whenever we need real distances.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


@dataclass
class GridSpec:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_km: float = 4.0

    @property
    def lat_step_deg(self) -> float:
        return self.resolution_km / 111.32  # 1 deg latitude ~= 111.32 km

    @property
    def lon_step_deg(self) -> float:
        mid_lat = 0.5 * (self.lat_min + self.lat_max)
        return self.resolution_km / (111.32 * np.cos(np.radians(mid_lat)))

    def build(self) -> pd.DataFrame:
        """Return a DataFrame of all grid-cell centers with columns
        ``cell_id, row, col, lat, lon``.

        Grid is laid out on a regular lat/lon mesh — at this resolution the
        Earth-curvature error vs an equal-area projection is well under
        the 4 km cell size.
        """
        lats = np.arange(self.lat_min, self.lat_max, self.lat_step_deg)
        lons = np.arange(self.lon_min, self.lon_max, self.lon_step_deg)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        n_rows, n_cols = lat_grid.shape
        rows = np.repeat(np.arange(n_rows), n_cols)
        cols = np.tile(np.arange(n_cols), n_rows)
        df = pd.DataFrame(
            {
                "cell_id": np.arange(n_rows * n_cols, dtype=np.int64),
                "row": rows.astype(np.int32),
                "col": cols.astype(np.int32),
                "lat": lat_grid.ravel().astype(np.float32),
                "lon": lon_grid.ravel().astype(np.float32),
            }
        )
        return df


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Inputs may be scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = [np.radians(np.asarray(x, dtype=np.float64))
                              for x in (lat1, lon1, lat2, lon2)]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def load_or_build_grid(processed_dir: Path, spec: GridSpec) -> pd.DataFrame:
    path = processed_dir / "grid_coordinates.csv"
    if path.exists():
        return pd.read_csv(path)
    df = spec.build()
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
