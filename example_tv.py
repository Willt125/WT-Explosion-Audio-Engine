"""Example demonstrating time-varying explosion synthesis with motion."""

from explosion import (
    AtmosphereWT,
    Geometry,
    LinearMotion,
    Ordnance,
    Rendering,
    save_wav,
    synthesize_explosion,
)


if __name__ == "__main__":
    ordnance = Ordnance(filler_mass=10.0, re=1.0, height_of_burst=0.0)
    src_motion = LinearMotion((-300.0, 0.0, 50.0), (200.0, 0.0, 0.0))
    geometry = Geometry(source=src_motion, receiver=(0.0, 0.0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=96000, pad=1.0)
    wave = synthesize_explosion(ordnance, geometry, atmos, render)
    save_wav("example_tv.wav", render.sample_rate, wave)
    print("Wrote example_tv.wav")
