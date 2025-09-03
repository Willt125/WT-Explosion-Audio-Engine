"""Explosion audio synthesis module following War Thunder atmosphere model.

This module provides utilities to synthesise pressure waveforms for outdoor
explosions such as bombs and HE shells.  The implementation is intentionally
compact and emphasises clarity over absolute physical fidelity.  Only ``numpy``
and ``scipy`` are required.

High level usage::

    from explosion import Ordnance, Geometry, AtmosphereWT, Rendering
    from explosion import synthesize_explosion, save_wav

    ordnance = Ordnance(filler_mass=10.0, re=1.0, height_of_burst=0.0)
    geometry = Geometry(source=(0, 0, 1.0), receiver=(100.0, 0, 1.7))
    atmos    = AtmosphereWT(rh=0.5)
    render   = Rendering(sample_rate=96000, pad=0.5)
    wave = synthesize_explosion(ordnance, geometry, atmos, render)
    save_wav("demo.wav", render.sample_rate, wave)

Two convenience helpers are provided:
``synthesize_explosion`` exposes full control over all parameters whereas
``synthesize_simple`` provides a minimal interface for common scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np
from scipy.io import wavfile

__all__ = [
    "Ordnance",
    "Geometry",
    "AtmosphereWT",
    "Rendering",
    "synthesize_explosion",
    "synthesize_simple",
    "save_wav",
]


# ---------------------------------------------------------------------------
# Dataclasses describing the problem
# ---------------------------------------------------------------------------


@dataclass
class Ordnance:
    """Explosive properties.

    Parameters
    ----------
    filler_mass : float
        Mass of explosive filler in kilograms.
    re : float, optional
        Relative effectiveness factor.  Default is 1.0 (TNT).
    height_of_burst : float, optional
        Height of burst above ground in metres.  Used only for the empirical
        height-of-burst boost to peak overpressure.
    """

    filler_mass: float
    re: float = 1.0
    height_of_burst: float = 0.0

    @property
    def tnt_equivalent(self) -> float:
        """Return the TNT equivalent mass (kg)."""
        return self.filler_mass * self.re


@dataclass
class Geometry:
    """Spatial configuration of source, receiver and ground properties."""

    source: Tuple[float, float, float]
    receiver: Tuple[float, float, float]
    flow_resistivity: float = 1e7  # Pa·s/m², high => rigid ground


@dataclass
class AtmosphereWT:
    """War Thunder like atmosphere description.

    Parameters
    ----------
    rh : float
        Relative humidity in ``[0, 1]``.
    """

    rh: float = 0.5

    # Polynomial coefficients for density model (ascending powers of h)
    H_MAX: float = 18300.0
    STD_RO0: float = 1.225
    RHO_COEFFS: Tuple[float, ...] = (
        1.0,
        -9.59387e-05,
        3.53118e-09,
        -5.83556e-14,
        2.28719e-19,
    )

    GAMMA: float = 1.4
    R: float = 287.05

    def density(self, h: np.ndarray | float) -> np.ndarray | float:
        """Return air density ``rho`` at altitude ``h`` (metres)."""
        h = np.asarray(h)
        h_clamped = np.clip(h, 0.0, self.H_MAX)
        # Polynomial with ascending powers
        poly = sum(c * h_clamped ** i for i, c in enumerate(self.RHO_COEFFS))
        return self.STD_RO0 * poly

    def speed_of_sound(self, h: np.ndarray | float) -> np.ndarray | float:
        """Return speed of sound (m/s) at altitude ``h`` using ISA surrogate."""
        h_km = np.asarray(h) / 1000.0
        return 340.0 - 0.4 * h_km - 0.02 * h_km**2

    def temperature(self, h: np.ndarray | float) -> np.ndarray | float:
        """Temperature in Kelvin derived from the local speed of sound."""
        c = self.speed_of_sound(h)
        return c**2 / (self.GAMMA * self.R)

    def pressure(self, h: np.ndarray | float) -> np.ndarray | float:
        """Ambient pressure in Pascals."""
        rho = self.density(h)
        T = self.temperature(h)
        return rho * self.R * T


@dataclass
class Rendering:
    """Rendering options."""

    sample_rate: int = 96_000
    pad: float = 0.5  # seconds of trailing silence
    target_peak_pa: float | None = None


# ---------------------------------------------------------------------------
# Helper mathematics
# ---------------------------------------------------------------------------


def _path_segments(src: np.ndarray, rec: np.ndarray, max_len: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return mid-point altitudes and segment lengths along the straight path."""
    vec = rec - src
    dist = np.linalg.norm(vec)
    n_seg = max(1, int(np.ceil(dist / max_len)))
    # Segment lengths are equal
    dl = dist / n_seg
    t_mid = (np.arange(n_seg) + 0.5) / n_seg
    mid_pts = src + np.outer(t_mid, vec)
    alt = np.abs(mid_pts[:, 2])  # absolute altitude for ground mirrored paths
    return alt, np.full(n_seg, dl)


def _atm_absorption(freq: np.ndarray, T: float, P: float, RH: float) -> np.ndarray:
    """ISO 9613-1 atmospheric absorption coefficient (Nepers/m).

    Parameters are scalar but ``freq`` can be a vector.  The implementation is
    a direct transcription of the standard's equations and is sufficient for
    realistic rendering purposes.
    """

    T_k = T
    T0 = 293.15
    P0 = 101325.0
    Psat = P0 * 10 ** (4.6151 - 6.8346 * (273.16 / T_k) ** 1.261)
    H = RH * Psat / P

    f_rel_O = P / P0 * (24.0 + 4.04e4 * H * (0.02 + H) / (0.391 + H))
    f_rel_N = (
        P / P0
        * (T_k / T0) ** -0.5
        * (9.0 + 280.0 * H * np.exp(-4.170 * ((T_k / T0) ** -1.0 / 3.0 - 1.0)))
    )

    term1 = 1.84e-11 * (P0 / P) * (T_k / T0) ** 0.5
    term2 = (
        (T_k / T0) ** -2.5
        * (
            0.01275 * np.exp(-2239.1 / T_k) / (f_rel_O + freq ** 2 / f_rel_O)
            + 0.1068 * np.exp(-3352.0 / T_k) / (f_rel_N + freq ** 2 / f_rel_N)
        )
    )
    alpha = 8.686 * freq ** 2 * (term1 + term2)  # dB / km
    alpha /= 1000.0  # dB / m
    # Convert from dB to Nepers
    return alpha / (20.0 * np.log10(np.e))


def _compute_transfer(
    src: np.ndarray,
    rec: np.ndarray,
    atmos: AtmosphereWT,
    freqs: np.ndarray,
    rh: float,
) -> Tuple[float, np.ndarray]:
    """Return delay and absorption along path in frequency domain."""

    alt, dl = _path_segments(src, rec)
    delay = np.sum(dl / atmos.speed_of_sound(alt))
    alpha = np.zeros_like(freqs)
    for h, d in zip(alt, dl):
        T = atmos.temperature(h)
        P = atmos.pressure(h)
        alpha += _atm_absorption(freqs, T, P, rh) * d
    return delay, alpha


def _reflection_coefficient(
    freq: np.ndarray, sigma: float, theta: float
) -> np.ndarray:
    """Delany–Bazley reflection coefficient for locally reacting ground."""

    # Guard against zero frequency which would otherwise lead to infinities.
    freq = np.asarray(freq, dtype=float)
    freq = np.maximum(freq, 1e-6)

    # Normalised specific impedance (Delany–Bazley 1970)
    X = freq / sigma
    Z = 1.0 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732
    return (Z * np.cos(theta) - 1.0) / (Z * np.cos(theta) + 1.0)


# ---------------------------------------------------------------------------
# Source wave generation
# ---------------------------------------------------------------------------


def _source_waveform(ord: Ordnance, rendering: Rendering, range_m: float) -> Tuple[np.ndarray, float, float]:
    """Create the two sided modified Friedlander source waveform.

    Returns
    -------
    wave : ndarray
        The source pressure waveform in Pascals.
    t_plus : float
        Positive phase duration.
    delay0 : float
        Start time of the waveform (always zero, kept for API symmetry).
    """

    W = ord.tnt_equivalent
    # scaled distance
    Z = max(range_m, 1e-6) / (W ** (1.0 / 3.0))

    # Peak overpressure (psi -> Pa)
    p_peak = 808.0 / Z ** 3 + 114.0 / Z ** 2 + 10.4 / Z
    p_peak *= 6894.757  # psi to Pa

    # Height-of-burst boost
    Hs = ord.height_of_burst / (W ** (1.0 / 3.0)) if W > 0 else 0.0
    B = 1.0 + 0.45 * np.exp(-Hs / 0.8)
    dp = p_peak * B

    # Positive phase duration
    c0 = 340.0  # use sea level value for estimate
    t_plus = (W ** (1.0 / 3.0)) * (0.02 + 0.12 * Z + 0.002 * Z ** 2) / c0
    t_plus = max(t_plus, 0.002)

    I_plus = 0.6 * dp * t_plus
    alpha = np.clip(dp * t_plus / I_plus, 0.8, 2.0)

    kappa = 1.5 + 2.0 * (1.0 - np.exp(-Z / 10.0))
    t_minus = kappa * t_plus
    eta = np.clip(0.25 + 0.35 * np.exp(-Z / 8.0), 0.2, 0.6)
    I_minus = eta * I_plus
    beta = I_minus / (dp * t_minus)

    duration = t_plus + t_minus
    n = int(np.ceil(duration * rendering.sample_rate))
    t = np.arange(n) / rendering.sample_rate
    wave = np.zeros_like(t)

    pos = t <= t_plus
    wave[pos] = dp * (1 - t[pos] / t_plus) * np.exp(-alpha * t[pos] / t_plus)

    neg = t > t_plus
    tau = t[neg] - t_plus
    wave[neg] = -beta * dp * (tau / t_minus) * np.exp(-tau / t_minus)

    return wave, t_plus, 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_explosion(
    ord: Ordnance,
    geo: Geometry,
    atmos: AtmosphereWT,
    rendering: Rendering,
) -> np.ndarray:
    """Synthesise an outdoor explosion pressure waveform.

    The function returns an array of pressure values in Pascals sampled at
    ``rendering.sample_rate``.
    """

    src = np.asarray(geo.source, dtype=float)
    rec = np.asarray(geo.receiver, dtype=float)
    d_direct = np.linalg.norm(rec - src)

    wave_src, t_plus, _ = _source_waveform(ord, rendering, d_direct)

    # Length needed after propagation
    d_img = np.linalg.norm(rec - np.array([src[0], src[1], -src[2]]))
    max_delay = max(d_direct, d_img) / atmos.speed_of_sound(0)
    n_fft = int(
        2 ** np.ceil(
            np.log2(
                len(wave_src)
                + int((max_delay + 8 * t_plus + rendering.pad) * rendering.sample_rate)
            )
        )
    )

    freqs = np.fft.rfftfreq(n_fft, 1.0 / rendering.sample_rate)
    src_spec = np.fft.rfft(wave_src, n_fft)

    # Direct path transfer
    delay_d, alpha_d = _compute_transfer(src, rec, atmos, freqs, atmos.rh)
    H_d = (1.0 / d_direct) * np.exp(-alpha_d) * np.exp(-1j * 2 * np.pi * freqs * delay_d)

    # Ground reflected path
    src_img = np.array([src[0], src[1], -src[2]])
    delay_g, alpha_g = _compute_transfer(src_img, rec, atmos, freqs, atmos.rh)

    # Incidence angle for reflection coefficient
    if src[2] > 0:
        # reflection point via image method
        t_ref = src[2] / (src[2] + rec[2]) if (src[2] + rec[2]) != 0 else 0.5
        ref_pt = src_img + t_ref * (rec - src_img)
        vec_inc = ref_pt - src
        cos_theta = abs(vec_inc[2]) / np.linalg.norm(vec_inc)
        theta = np.arccos(np.clip(cos_theta, 0.0, 1.0))
    else:
        theta = 0.0

    Rg = _reflection_coefficient(freqs, geo.flow_resistivity, theta)
    H_g = (
        Rg
        * (1.0 / d_img)
        * np.exp(-alpha_g)
        * np.exp(-1j * 2 * np.pi * freqs * delay_g)
    )

    total_spec = src_spec * (H_d + H_g)
    wave = np.fft.irfft(total_spec, n_fft)

    # Trim to relevant portion
    n_out = int((max(delay_d, delay_g) + 8 * t_plus + rendering.pad) * rendering.sample_rate)
    wave = wave[:n_out]

    # Optional calibration
    if rendering.target_peak_pa is not None and np.max(np.abs(wave)) > 0:
        idx = int(delay_d * rendering.sample_rate)
        search = wave[idx : idx + int(0.02 * rendering.sample_rate)]
        peak = np.max(np.abs(search))
        if peak > 0:
            wave *= rendering.target_peak_pa / peak

    return wave


def synthesize_simple(
    filler_mass: float,
    range_m: float,
    src_height: float = 1.0,
    rec_height: float = 1.7,
    re: float = 1.0,
    hob: float = 0.0,
    sample_rate: int = 96_000,
    pad: float = 0.5,
    rh: float = 0.5,
    sigma: float = 1e7,
) -> Tuple[np.ndarray, int]:
    """Convenience wrapper for common use cases.

    Returns the pressure waveform and sample rate.
    """

    ord = Ordnance(filler_mass, re, hob)
    geo = Geometry((0.0, 0.0, src_height), (range_m, 0.0, rec_height), sigma)
    atmos = AtmosphereWT(rh)
    render = Rendering(sample_rate, pad)
    wave = synthesize_explosion(ord, geo, atmos, render)
    return wave, render.sample_rate


def save_wav(path: str, sample_rate: int, data: Iterable[float]) -> None:
    """Save ``data`` to ``path`` as a floating point WAV file."""
    wavfile.write(path, sample_rate, np.asarray(data, dtype=np.float32))
