from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class RayHit:
    """Result of a raycast query in ENU coordinates.

    Parameters are in metres with the ``normal`` vector being unit length.
    ``distance`` is the segment distance to the hit point or ``np.inf`` if
    no intersection occurred.
    """

    hit: bool
    point: Vec3
    normal: Vec3
    material_id: int
    distance: float


@dataclass(frozen=True)
class Plane:
    """Infinite plane specified by a point and unit normal (Z-up)."""

    point: Vec3
    normal: Vec3


class AcousticWorld(Protocol):
    """Minimal interface for terrain and material queries.

    All coordinates follow an ENU, right handed, metres, Z-up convention.
    """

    def elevation(self, x: float, y: float) -> float:
        """Return ground elevation ``z`` at ``(x, y)`` in metres."""

    def surface_normal(self, x: float, y: float) -> Vec3:
        """Return unit surface normal at ``(x, y)`` (default ``(0,0,1)``)."""

    def material_at(self, x: float, y: float) -> int:
        """Return material identifier at ``(x, y)``."""

    def impedance_at(self, x: float, y: float) -> float:
        """Return flow resistivity (Pa·s/m²) of material at ``(x, y)``."""

    def raycast(self, p0: Vec3, p1: Vec3) -> RayHit:
        """Cast a segment from ``p0`` to ``p1`` and return the first hit."""

    def image_surface_candidates(
        self, src: Vec3, rec: Vec3, max_order: int = 1
    ) -> Sequence[Plane]:
        """Return planes to test for image source reflections."""

    def nearest_edges(self, p: Vec3, radius: float) -> Sequence[int]:
        """Placeholder for diffraction queries (currently empty)."""
