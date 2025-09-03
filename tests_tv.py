import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter1d

from explosion import (
    AtmosphereWT,
    Geometry,
    LinearMotion,
    Ordnance,
    Rendering,
    synthesize_explosion,
    synthesize_explosion_tv,
    solve_retarded_time,
)


def test_static_regression():
    ord = Ordnance(10.0, re=1.0, height_of_burst=0.0)
    geo = Geometry(source=(0.0, 0.0, 1.0), receiver=(100.0, 0.0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=8000, pad=0.2)
    w_static = synthesize_explosion(ord, geo, atmos, render)
    w_tv = synthesize_explosion_tv(ord, geo, atmos, render)
    idx = np.argmax(np.abs(w_static))
    n = int(0.2 * render.sample_rate)
    win = slice(idx, idx + n)
    err = np.sqrt(np.mean((w_static[win] - w_tv[win]) ** 2))
    ref = np.sqrt(np.mean(w_static[win] ** 2))
    assert err < 1e-2 * ref


def test_doppler_glide():
    ord = Ordnance(10.0)
    src = LinearMotion((-300.0, 0.0, 50.0), (200.0, 0.0, 0.0))
    geo = Geometry(source=src, receiver=(0.0, 0.0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=8000, pad=0.2)
    y = synthesize_explosion(ord, geo, atmos, render)
    f, t, S = spectrogram(y, fs=render.sample_rate, nperseg=512, noverlap=256)
    ridge = f[np.argmax(S, axis=0)]
    assert ridge.max() > ridge.min()
    os.makedirs('rendered', exist_ok=True)
    plt.figure()
    plt.pcolormesh(t, f, 20 * np.log10(S + 1e-12), shading='gouraud')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig('rendered/doppler.png')
    plt.close()


def test_energy_falloff():
    ord = Ordnance(10.0)
    src = (0.0, 0.0, 1.0)
    rec_motion = LinearMotion((50.0, 0.0, 1.7), (100.0, 0.0, 0.0))
    geo = Geometry(source=src, receiver=rec_motion)
    atmos = AtmosphereWT(rh=0.0)
    render = Rendering(sample_rate=8000, pad=0.2)
    import explosion

    orig_design = explosion.design_absorption_fir
    explosion.design_absorption_fir = lambda D_ref, atmos, sr, taps=512: np.array([1.0])
    y = synthesize_explosion(ord, geo, atmos, render)
    explosion.design_absorption_fir = orig_design

    sr = render.sample_rate
    env = maximum_filter1d(np.abs(y), size=sr // 20)
    Ds = np.array([solve_retarded_time(geo.src_fn, geo.rec_fn, n / sr, atmos)[1] for n in range(len(y))])
    mask = (Ds > 50) & (Ds < 150) & (env > 1e-3)
    ratio = env[mask] * Ds[mask]
    rel = np.std(ratio) / np.mean(ratio)
    assert np.isfinite(rel)


def test_ground_image_on_off():
    ord = Ordnance(10.0)
    src = (0.0, 0.0, 1.0)
    rec = (100.0, 0.0, 1.7)
    geo_soft = Geometry(source=src, receiver=rec, flow_resistivity=5e4)
    geo_rigid = Geometry(source=src, receiver=rec, flow_resistivity=1e12)
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=8000, pad=0.2)
    y_soft = synthesize_explosion_tv(ord, geo_soft, atmos, render)
    y_rigid = synthesize_explosion_tv(ord, geo_rigid, atmos, render)
    idx = np.argmax(np.abs(y_rigid))
    win = slice(idx + int(0.01 * render.sample_rate), idx + int(0.05 * render.sample_rate))
    rms_soft = np.sqrt(np.mean(y_soft[win] ** 2))
    rms_rigid = np.sqrt(np.mean(y_rigid[win] ** 2))
    assert rms_soft < rms_rigid
    os.makedirs('rendered', exist_ok=True)
    t = np.arange(len(y_soft)) / render.sample_rate
    plt.figure()
    plt.plot(t, y_soft, label='sigma=5e4')
    plt.plot(t, y_rigid, label='sigma=inf')
    plt.xlim((idx / render.sample_rate - 0.01, idx / render.sample_rate + 0.1))
    plt.legend()
    plt.savefig('rendered/ground.png')
    plt.close()


if __name__ == '__main__':
    test_static_regression()
    test_doppler_glide()
    test_energy_falloff()
    test_ground_image_on_off()
    print('tests_tv passed')
