"""UTM projection and pixel transform for the offline renderer.

The printed map is UTM-aligned (grid north up), so projecting lat/lon to
UTM meter coordinates is all the geometry we need: pixels are a linear map
of UTM meters at ``dpi / 0.0254`` px per meter.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from schemas.utm import UTMBoundingBox

# WGS84
_A = 6378137.0
_E2 = 0.0066943799901413165
_K0 = 0.9996


def _utm_from_latlon(lat: np.ndarray, lon: np.ndarray, zone_number: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized WGS84 -> UTM (standard transverse mercator series)."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lon0 = math.radians((zone_number - 1) * 6 - 180 + 3)

    ep2 = _E2 / (1.0 - _E2)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    tan_lat = np.tan(lat)

    n = _A / np.sqrt(1.0 - _E2 * sin_lat ** 2)
    t = tan_lat ** 2
    c = ep2 * cos_lat ** 2
    a = cos_lat * (lon - lon0)

    m = _A * (
        (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * lat
        - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * np.sin(2 * lat)
        + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * np.sin(4 * lat)
        - (35 * _E2 ** 3 / 3072) * np.sin(6 * lat)
    )

    easting = _K0 * n * (
        a
        + (1 - t + c) * a ** 3 / 6
        + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * a ** 5 / 120
    ) + 500000.0

    northing = _K0 * (
        m
        + n * tan_lat * (
            a ** 2 / 2
            + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
            + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * a ** 6 / 720
        )
    )
    return easting, northing


class Projector:
    """Lat/lon -> canvas pixel transform for one map region."""

    def __init__(self, bbox: UTMBoundingBox, scale: Fraction, dpi: int) -> None:
        self.bbox = bbox
        self.scale = scale
        # paper pixels per ground meter: 1 m on paper = scale m on ground
        self.px_per_m = float(scale) * dpi / 0.0254
        self.width_m = float(bbox.right - bbox.left)
        self.height_m = float(bbox.top - bbox.bottom)
        self.width_px = int(round(self.width_m * self.px_per_m))
        self.height_px = int(round(self.height_m * self.px_per_m))

    def project(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Project lat/lon (scalar or array) to pixel x/y arrays."""
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        easting, northing = _utm_from_latlon(lat, lon, self.bbox.zone_number)
        x = (easting - self.bbox.left) * self.px_per_m
        y = (self.bbox.top - northing) * self.px_per_m
        return x, y

    def project_points(self, points: list[tuple[float, float]]) -> np.ndarray:
        """Project a list of (lat, lon) into an (N, 2) float array of pixels."""
        if not points:
            return np.zeros((0, 2), dtype=np.float64)
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        x, y = self.project(lats, lons)
        return np.column_stack((x, y))

    def corners_px(self) -> np.ndarray:
        """The four canvas corners in pixel coordinates (grid aligned)."""
        w, h = float(self.width_px), float(self.height_px)
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
