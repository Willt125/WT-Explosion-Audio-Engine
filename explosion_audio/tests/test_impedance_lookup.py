import numpy as np

from ..explosion import (
    AtmosphereWT,
    Geometry,
    Ordnance,
    Rendering,
    synthesize_explosion,
)
from ..world.flat import FlatWorld
from ..world.materials import default_materials


def reflection_amplitude(world: FlatWorld) -> float:
    ordn = Ordnance(filler_mass=1.0)
    geo = Geometry(source=(0.0, 0.0, 1.0), receiver=(100.0, 0.0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=8000, pad=0.2, use_burgers=False)
    wave = synthesize_explosion(ordn, geo, atmos, render, world)
    c0 = float(atmos.speed_of_sound(0.0))
    src = np.array(geo.source, float)
    rec = np.array(geo.receiver, float)
    src_img = np.array([src[0], src[1], -src[2]])
    delay_g = np.linalg.norm(rec - src_img) / c0
    idx = int(delay_g * render.sample_rate)
    return float(np.max(np.abs(wave[idx - 5 : idx + 5])))


def test_reflection_magnitude_monotonic() -> None:
    mats = default_materials()
    pairs = sorted(
        ((mid, mat.flow_resistivity) for mid, mat in mats.materials.items()),
        key=lambda m: m[1],
    )
    amps = []
    for mid, _sigma in pairs:
        world = FlatWorld(z0=0.0, ground_material_id=mid, materials=mats)
        amps.append(reflection_amplitude(world))
    assert amps == sorted(amps)
