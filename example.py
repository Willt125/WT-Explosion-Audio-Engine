"""Render example explosion waveforms and save to WAV files."""

from pathlib import Path

from explosion import (
    Ordnance,
    Geometry,
    AtmosphereWT,
    Rendering,
    synthesize_explosion,
    save_wav,
)


def main() -> None:
    out_dir = Path("rendered")
    out_dir.mkdir(exist_ok=True)

    # Example 1: 10 kg TNT equivalent at 100 m range
    ord1 = Ordnance(filler_mass=10.0, re=1.0, height_of_burst=0.0)
    geo1 = Geometry(source=(0, 0, 1.0), receiver=(100.0, 0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=96_000, pad=0.5)
    wave = synthesize_explosion(ord1, geo1, atmos, render)
    save_wav(out_dir / "bomb_10kg_100m.wav", render.sample_rate, wave)

    # Example 2: 1 kg charge at 20 m with slight height-of-burst
    ord2 = Ordnance(filler_mass=1.0, re=1.0, height_of_burst=2.0)
    geo2 = Geometry(source=(0, 0, 2.0), receiver=(20.0, 0, 1.7))
    wave2 = synthesize_explosion(ord2, geo2, atmos, render)
    save_wav(out_dir / "shell_1kg_20m.wav", render.sample_rate, wave2)


if __name__ == "__main__":
    main()
