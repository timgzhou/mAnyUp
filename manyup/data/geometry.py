"""Where a raster sits on the ground, carried through every crop and resample.

`transform` is the GDAL/rasterio affine (a, b, c, d, e, f): x = a*col + b*row + c,
y = d*col + e*row + f, in `crs` units. A dataset without real georeferencing uses
Geometry.unknown(): crs/transform are None rather than a placeholder that looks real.
"""
from __future__ import annotations

from dataclasses import dataclass

Affine = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Geometry:
    crs: str | None          # e.g. "EPSG:2154"; None = location unknown
    transform: Affine | None
    shape: tuple[int, int]   # (H, W) in pixels

    @staticmethod
    def unknown(shape: tuple[int, int]) -> "Geometry":
        return Geometry(crs=None, transform=None, shape=tuple(shape))

    @property
    def known(self) -> bool:
        return self.transform is not None

    @property
    def gsd(self) -> tuple[float, float] | None:
        """(x, y) pixel size in crs units; None if unknown or rotated."""
        if not self.known:
            return None
        a, b, _, d, e, _ = self.transform
        return (abs(a), abs(e)) if b == 0 and d == 0 else None

    def crop(self, row: int, col: int, h: int, w: int) -> "Geometry":
        H, W = self.shape
        if not (0 <= row and 0 <= col and row + h <= H and col + w <= W):
            raise ValueError(f"crop ({row},{col},{h},{w}) outside raster {self.shape}")
        if not self.known:
            return Geometry.unknown((h, w))
        a, b, c, d, e, f = self.transform
        return Geometry(self.crs, (a, b, c + a * col + b * row, d, e, f + d * col + e * row),
                        (h, w))

    def rescale(self, shape: tuple[int, int]) -> "Geometry":
        """Same footprint on a (H', W') grid, e.g. a token grid or an upsampled map."""
        H, W = self.shape
        h, w = shape
        if not self.known:
            return Geometry.unknown((h, w))
        a, b, c, d, e, f = self.transform
        sx, sy = W / w, H / h
        return Geometry(self.crs, (a * sx, b * sy, c, d * sx, e * sy, f), (h, w))
