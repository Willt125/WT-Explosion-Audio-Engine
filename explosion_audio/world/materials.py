from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class Material:
    """Simple acoustic material description.

    Parameters
    ----------
    id : int
        Identifier used by the world queries.
    name : str
        Human readable name.
    flow_resistivity : float
        Flow resistivity (Pa·s/m²).
    absorption_1kHz_db : float, optional
        1 kHz absorption in decibels. Currently unused.
    """

    id: int
    name: str
    flow_resistivity: float
    absorption_1kHz_db: float = 0.0


@dataclass
class MaterialDB:
    """Dictionary style container for :class:`Material` objects."""

    materials: Dict[int, Material]

    def by_id(self, id: int) -> Material:
        """Return material with ``id``."""
        return self.materials[id]


def default_materials() -> MaterialDB:
    """Return a default material dictionary."""

    mats = {
        0: Material(0, "default", 5e4),
        1: Material(1, "grass/soil", 2e4),
        2: Material(2, "asphalt/concrete", 1e7),
        3: Material(3, "water", 5e6),
    }
    return MaterialDB(mats)
