from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .api import AcousticWorld, Plane, RayHit, Vec3
from .flat import FlatWorld
from .materials import MaterialDB, default_materials


@dataclass
class TileStreamer(AcousticWorld):
    """Stubbed tile streamer that answers the :class:`AcousticWorld` API."""

    tile_size_m: float = 512.0
    materials: MaterialDB | None = None

    def __post_init__(self) -> None:
        self.materials = self.materials or default_materials()
        self._flat = FlatWorld(materials=self.materials)

    # Delegate all API calls to internal FlatWorld tile ----------------------
    def elevation(self, x: float, y: float) -> float:
        return self._flat.elevation(x, y)

    def surface_normal(self, x: float, y: float) -> Vec3:
        return self._flat.surface_normal(x, y)

    def material_at(self, x: float, y: float) -> int:
        return self._flat.material_at(x, y)

    def impedance_at(self, x: float, y: float) -> float:
        return self._flat.impedance_at(x, y)

    def raycast(self, p0: Vec3, p1: Vec3) -> RayHit:
        return self._flat.raycast(p0, p1)

    def image_surface_candidates(self, src: Vec3, rec: Vec3, max_order: int = 1) -> Sequence[Plane]:
        return self._flat.image_surface_candidates(src, rec, max_order)

    def nearest_edges(self, p: Vec3, radius: float) -> Sequence[int]:
        return []
