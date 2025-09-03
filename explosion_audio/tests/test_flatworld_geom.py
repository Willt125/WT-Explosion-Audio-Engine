import numpy as np

from ..world.api import Plane
from ..world.flat import FlatWorld


def test_ground_intersection() -> None:
    world = FlatWorld(z0=0.0)
    hit = world.raycast((0.0, 0.0, 10.0), (10.0, 0.0, -10.0))
    assert hit.hit
    assert np.allclose(hit.point, (5.0, 0.0, 0.0))
    assert np.allclose(hit.normal, (0.0, 0.0, 1.0))


def test_building_plane_precedence() -> None:
    plane = Plane((50.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    world = FlatWorld(z0=0.0, building_plane=plane)
    hit = world.raycast((0.0, 0.0, 1.0), (100.0, 0.0, 1.0))
    assert hit.hit
    assert np.allclose(hit.point, (50.0, 0.0, 1.0))
    assert np.allclose(hit.normal, (-1.0, 0.0, 0.0))
