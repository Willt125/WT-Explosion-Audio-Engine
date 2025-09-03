from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import numpy as np

from .api import AcousticWorld, Plane, RayHit, Vec3
from .materials import MaterialDB, default_materials


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _ray_plane(p0: np.ndarray, p1: np.ndarray, plane: Plane) -> Tuple[Optional[np.ndarray], float]:
    """Return intersection point and distance along segment or ``(None, inf)``."""
    n = np.asarray(plane.normal, float)
    n = _normalize(n)
    denom = np.dot(n, p1 - p0)
    if abs(denom) < 1e-12:
        return None, np.inf
    t = np.dot(n, np.asarray(plane.point) - p0) / denom
    if t < 0.0 or t > 1.0:
        return None, np.inf
    point = p0 + t * (p1 - p0)
    dist = t * np.linalg.norm(p1 - p0)
    return point, dist


@dataclass
class FlatWorld(AcousticWorld):
    """Infinite flat ground with optional vertical plane.

    Parameters
    ----------
    z0 : float, optional
        Ground plane height in metres.
    ground_material_id : int, optional
        Material id for ground plane.
    building_plane : Plane, optional
        Optional second plane (e.g. building facade).
    building_material_id : int, optional
        Material id for ``building_plane``.
    materials : MaterialDB, optional
        Material database used for impedance lookups.
    """

    z0: float = 0.0
    ground_material_id: int = 2
    building_plane: Plane | None = None
    building_material_id: int = 2
    materials: MaterialDB | None = None

    def __post_init__(self) -> None:
        self.materials = self.materials or default_materials()
        if self.building_plane is not None:
            n = np.asarray(self.building_plane.normal, float)
            n = _normalize(n)
            self.building_plane = Plane(tuple(self.building_plane.point), tuple(n))

    # -- AcousticWorld interface -------------------------------------------------
    def elevation(self, x: float, y: float) -> float:
        return self.z0

    def surface_normal(self, x: float, y: float) -> Vec3:
        return (0.0, 0.0, 1.0)

    def material_at(self, x: float, y: float) -> int:
        return self.ground_material_id

    def impedance_at(self, x: float, y: float) -> float:
        return self.materials.by_id(self.material_at(x, y)).flow_resistivity

    def raycast(self, p0: Vec3, p1: Vec3) -> RayHit:
        a = np.asarray(p0, float)
        b = np.asarray(p1, float)
        planes: list[Tuple[Plane, int]] = [
            (Plane((0.0, 0.0, self.z0), (0.0, 0.0, 1.0)), self.ground_material_id)
        ]
        if self.building_plane is not None:
            planes.append((self.building_plane, self.building_material_id))
        best_dist = np.inf
        best: Tuple[Plane, int, np.ndarray] | None = None
        for pl, mid in planes:
            point, dist = _ray_plane(a, b, pl)
            if point is not None and dist < best_dist:
                best_dist = dist
                best = (pl, mid, point)
        if best is None:
            return RayHit(False, p1, (0.0, 0.0, 1.0), self.ground_material_id, np.inf)
        pl, mid, pt = best
        n = _normalize(np.asarray(pl.normal, float))
        return RayHit(True, tuple(pt), tuple(n), mid, float(best_dist))

    def image_surface_candidates(self, src: Vec3, rec: Vec3, max_order: int = 1) -> Sequence[Plane]:
        planes = [Plane((0.0, 0.0, self.z0), (0.0, 0.0, 1.0))]
        if self.building_plane is not None:
            planes.append(self.building_plane)
        return planes

    def nearest_edges(self, p: Vec3, radius: float) -> Sequence[int]:
        return []
