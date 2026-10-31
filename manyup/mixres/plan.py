"""A tokenization plan: which (sample, month, window) get tokenized at a finer patch size on top
of the uniform season tokens. Selectors return one; the model encodes whatever it is given."""
from dataclasses import dataclass

import torch

LOOKUP_PS = 4       # the selection grid: 64 px tiles as 16 x 16 cells of 4 px
TILE = 64


@dataclass
class Plan:
    """N crops, each one month (or one raw acquisition) of one sample at `patch` px.

    b, t      (N,) long   sample, month slot (the decoder routes by it)
    y, x      (N,) long   the picked cell on the 16 x 16 ps4 grid (the decoder routes by it)
    py, px    (N,) long   top-left pixel of the window actually cropped
    patch     int         window side in px (every crop of a plan shares it)
    src       "months": cut from the monthly composites; else a modality name: cut from that
              modality's raw acquisitions (Batch.dates), tokens marked with the model's
              date_type embedding
    src_t     (N,) long   the step cropped in the source (None: t)
    slot      (N,) float  time slot of each crop (None: its month's season slot)
    """
    b: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    py: torch.Tensor
    px: torch.Tensor
    patch: int
    src: str = "months"
    src_t: torch.Tensor | None = None
    slot: torch.Tensor | None = None
    offgrid: bool = False       # windows at any pixel (Quadrant offgrid): never routed per cell

    def __len__(self) -> int:
        return len(self.b)

    @property
    def on_grid(self) -> bool:
        """Windows are exactly the picked ps4 cells: the decoder pools per cell, not per pixel."""
        return self.patch == LOOKUP_PS and not self.offgrid

    @property
    def cells(self) -> int:
        """Footprint side in ps4 cells (a ps2 crop: part of one)."""
        return max(1, self.patch // LOOKUP_PS)

    def split(self, patch: int) -> "Plan":
        """Each window as its (self.patch / patch)^2 sub-windows at `patch` px, consecutive (a
        crop's i-th sub-window at row-major index i); b, t, y, x, src_t, slot repeated."""
        f = self.patch // patch
        if f < 2 or self.patch % patch:
            raise ValueError(f"split {self.patch} px windows into {patch} px: not a divisor")
        k = f * f
        r = lambda v: None if v is None else v.repeat_interleave(k)              # noqa: E731
        o = torch.arange(f, device=self.b.device) * patch
        dy, dx = (o.repeat_interleave(f).repeat(len(self)), o.repeat(f).repeat(len(self)))
        return Plan(r(self.b), r(self.t), r(self.y), r(self.x), r(self.py) + dy, r(self.px) + dx,
                    patch, self.src, r(self.src_t), r(self.slot), self.offgrid)

    def shifted(self, d: int) -> "Plan":
        """Windows moved by d px (a padded frame's grid -> the tile), clamped into the tile."""
        py, px = ((v + d).clamp(0, TILE - self.patch) for v in (self.py, self.px))
        return Plan(self.b, self.t, py // LOOKUP_PS, px // LOOKUP_PS, py, px, self.patch,
                    self.src, self.src_t, self.slot, self.offgrid)

    @classmethod
    def from_grid(cls, mask: torch.Tensor, patch: int) -> "Plan":
        """mask (B, T, 64 / patch, 64 / patch) bool on the patch's own non-overlapping grid ->
        the plan; (y, x) = the ps4 cell under each window's top-left pixel."""
        b, t, gy, gx = mask.nonzero(as_tuple=True)
        py, px = gy * patch, gx * patch
        return cls(b, t, py // LOOKUP_PS, px // LOOKUP_PS, py, px, patch)

    @classmethod
    def empty(cls, patch: int, device) -> "Plan":
        z = torch.zeros(0, dtype=torch.long, device=device)
        return cls(z, z, z, z, z, z, patch)

    @classmethod
    def from_mask(cls, mask: torch.Tensor, patch: int, loc_px: torch.Tensor | None = None
                  ) -> "Plan":
        """mask (B, T, 16, 16) bool of picked (month, cell) -> the plan. A cell's window is
        loc_px[cell] when given (a candidate lattice, --loc_stride), else the patch-px window
        centred on the cell, clamped into the tile."""
        b, t, y, x = mask.nonzero(as_tuple=True)
        if loc_px is not None:
            lut = loc_px.to(mask.device)
            py, px = lut[y], lut[x]
        else:
            py, px = ((v * LOOKUP_PS + LOOKUP_PS // 2 - patch // 2).clamp(0, TILE - patch)
                      for v in (y, x))
        return cls(b, t, y, x, py, px, patch)
