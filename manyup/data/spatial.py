"""Spatial tiling: split each native sample into fixed-size tiles, each its own sample.

Tile ids are "<native id>@r<row>c<col>" (pixel offsets of the tile's top-left corner), so an
id says exactly which pixels it covers regardless of enumeration order. `order` fixes the
enumeration (and so dataset indices): "row-major" or "col-major" (top-left, bottom-left,
top-right, bottom-right for 2x2 -- the order OlmoEarth's PASTIS prep uses).
"""
from __future__ import annotations

from dataclasses import dataclass

ORDERS = ("row-major", "col-major")


@dataclass(frozen=True)
class Tiling:
    size: int
    order: str

    def __post_init__(self):
        if self.order not in ORDERS:
            raise ValueError(f"order {self.order!r} not in {ORDERS}")

    @property
    def key(self) -> str:
        return f"tile{self.size}"

    def offsets(self, shape: tuple[int, int]) -> list[tuple[int, int]]:
        H, W = shape
        if H % self.size or W % self.size:
            raise ValueError(f"tile {self.size} does not divide native shape {shape}")
        rows, cols = range(0, H, self.size), range(0, W, self.size)
        if self.order == "row-major":
            return [(r, c) for r in rows for c in cols]
        return [(r, c) for c in cols for r in rows]

    @staticmethod
    def tile_id(native_id: str, row: int, col: int) -> str:
        return f"{native_id}@r{row}c{col}"

    @staticmethod
    def parse(sample_id: str) -> tuple[str, int, int]:
        native, _, rc = sample_id.rpartition("@r")
        r, _, c = rc.partition("c")
        return native, int(r), int(c)
