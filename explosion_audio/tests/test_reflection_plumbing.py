import numpy as np

from ..explosion import (
    AtmosphereWT,
    Geometry,
    Ordnance,
    Rendering,
    synthesize_explosion,
)
from ..world.flat import FlatWorld


def test_direct_and_reflected_paths() -> None:
    world = FlatWorld(z0=0.0, ground_material_id=0)
    ordn = Ordnance(filler_mass=1.0)
    geo = Geometry(source=(0.0, 0.0, 1.0), receiver=(100.0, 0.0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=48000, pad=0.2, use_burgers=False)
    wave = synthesize_explosion(ordn, geo, atmos, render, world)
    c0 = float(atmos.speed_of_sound(0.0))
    src = np.array(geo.source, float)
    rec = np.array(geo.receiver, float)
    delay_d = np.linalg.norm(rec - src) / c0
    src_img = np.array([src[0], src[1], -src[2]])
    delay_g = np.linalg.norm(rec - src_img) / c0
    sr = render.sample_rate
    idx_d = int(delay_d * sr)
    idx_g = int(delay_g * sr)
    assert np.abs(wave[idx_d]) > 0
    assert np.abs(wave[idx_g]) > 0

    hit = world.raycast(tuple(src_img), tuple(rec))
    vec_inc = np.array(hit.point) - src
    cos_theta = np.abs(np.dot(vec_inc, hit.normal)) / np.linalg.norm(vec_inc)
    theta = np.arccos(np.clip(cos_theta, 0.0, 1.0))
    from ..explosion import _reflection_coefficient

    Rg = _reflection_coefficient(np.array([1000.0]), world.impedance_at(hit.point[0], hit.point[1]), theta)[0]
    assert abs(Rg) < 1.0
