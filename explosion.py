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
        h = np.asarray(h, dtype=float)
        h_clamped = np.clip(h, 0.0, self.H_MAX)
        # Polynomial with ascending powers times the H_MAX factor
        poly = sum(c * h_clamped ** i for i, c in enumerate(self.RHO_COEFFS))
        factor = self.H_MAX / np.maximum(self.H_MAX, np.maximum(h, 1e-9))
        return self.STD_RO0 * poly * factor

    def speed_of_sound(self, h: np.ndarray | float) -> np.ndarray | float:
        """Return speed of sound (m/s) at altitude ``h`` using ISA surrogate."""
        h_km = np.asarray(h) / 1000.0
        c = 340.0 - 0.4 * h_km - 0.02 * h_km**2
        return np.maximum(c, 300.0) # prevent pathological slow c at high h

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

def _sat_vapor_pressure_Pa(T_k: float) -> float:
    # Buck (over water), T in K
    T_c = T_k - 273.15
    return 611.21 * np.exp((18.678 - (T_c / 234.5)) * (T_c / (257.14 + T_c)))


def _atm_absorption(freq: np.ndarray, T_k: float, P_pa: float, RH: float) -> np.ndarray:
    f = np.maximum(np.asarray(freq, dtype=float), 1e-3)
    T0, P0 = 293.15, 101325.0
    Tr, Pr = T_k / T0, P_pa / P0

    Psat = _sat_vapor_pressure_Pa(T_k)
    h = RH * Psat / P_pa  # molar fraction

    frO2 = Pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN2 = Pr * Tr**(-0.5) * (9.0 + 280.0 * h * np.exp(-4.170 * (Tr**(-1.0/3.0) - 1.0)))

    term_class = 1.84e-11 * (P0 / P_pa) * np.sqrt(Tr)
    term_relax = Tr**(-2.5) * (
        0.01275 * np.exp(-2239.1 / T_k) / (frO2 + (f**2) / frO2)
        + 0.1068  * np.exp(-3352.0 / T_k) / (frN2 + (f**2) / frN2)
    )
    # α in Np/m
    alpha_np = (f**2) * (term_class + term_relax)
    return alpha_np


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
    sigma = max(float(sigma), 1.0) # guard

    # Normalised specific impedance (Delany–Bazley 1970)
    X = freq / sigma
    Z = 1.0 + 0.0571 * X ** -0.754 - 1j * 0.087 * X ** -0.732
    return (Z * np.cos(theta) - 1.0) / (Z * np.cos(theta) + 1.0)


# ---------------------------------------------------------------------------
# Source wave generation
# ---------------------------------------------------------------------------


def _source_waveform(ord: Ordnance, rendering: Rendering, range_m: float, gamma_neg: float = 1.0) -> Tuple[np.ndarray, float, float]:
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

    W = max(ord.tnt_equivalent, 1e-9)
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
    beta = (I_minus / (dp * t_minus)) * (gamma_neg**2)

    duration = t_plus + t_minus
    n = int(np.ceil(duration * rendering.sample_rate))
    t = np.arange(n) / rendering.sample_rate
    wave = np.zeros_like(t)

    pos = t <= t_plus
    wave[pos] = dp * (1 - t[pos] / t_plus) * np.exp(-alpha * t[pos] / t_plus)

    neg = t > t_plus
    tau = t[neg] - t_plus
    wave[neg] = -beta * dp * (tau / t_minus) * np.exp(-gamma_neg * tau / t_minus)

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
    
    # Image source defined up front (z mirrored across ground plane)
    src_img = np.array([src[0], src[1], -src[2]])
    
    d_direct = max(np.linalg.norm(rec - src), 1e-6)
    d_img = max(np.linalg.norm(rec - src_img), 1e-6)

    # Build source (range uses direct slant range)
    wave_src, t_plus, _ = _source_waveform(ord, rendering, d_direct)

    # First, get segmented delays (scalar freq just to compute delay quickly)
    delay_d, _ = _compute_transfer(src, rec, atmos, freqs=np.array([1.0]), rh=atmos.rh)
    delay_g, _ = _compute_transfer(src_img, rec, atmos, freqs=np.array([1.0]), rh=atmos.rh)
    max_delay = max(delay_d, delay_g)
    
    # FFT size
    n_needed = len(wave_src) + int((max_delay + 8 * t_plus + rendering.pad) * rendering.sample_rate)
    n_fft = int(1 << int(np.ceil(np.log2(max(8, n_needed)))))

    freqs = np.fft.rfftfreq(n_fft, 1.0 / rendering.sample_rate)
    src_spec = np.fft.rfft(wave_src, n_fft)

    # Recompute with full frequency vector to get absorption properly
    delay_d, alpha_d = _compute_transfer(src, rec, atmos, freqs, atmos.rh)
    delay_g, alpha_g = _compute_transfer(src_img, rec, atmos, freqs, atmos.rh)
    
    v = rec - src_img
    den = v[2]
    if abs(den) < 1e-9:
        t_ref = 0.5 # parallel; arbitrary midpoint
    else:
        t_ref = -src_img[2] / den # since plane z=0
    t_ref = np.clip(t_ref, 0.0, 1.0)
    ref_pt = src_img + t_ref * v
    inc_vec = ref_pt - src
    inc_norm = np.linalg.norm(inc_vec) + 1e-12
    cos_theta = abs(inc_vec[2]) / inc_norm
    theta = np.arccos(np.clip(cos_theta, 0.0, 1.0))

    Rg = _reflection_coefficient(freqs, geo.flow_resistivity, theta)
    
    H_d = (1.0 / d_direct) * np.exp(-alpha_d) * np.exp(-1j * 2 * np.pi * freqs * delay_d)
    H_g = Rg * (1.0 / d_img) * np.exp(-alpha_g) * np.exp(-1j * 2 * np.pi * freqs * delay_g)

    total_spec = src_spec * (H_d + H_g)
    wave = np.fft.irfft(total_spec, n_fft)

    # Trim to relevant portion
    n_out = int((max(delay_d, delay_g) + 8 * t_plus + rendering.pad) * rendering.sample_rate)
    wave = wave[:max(n_out, 1)]

    if rendering.target_peak_pa is not None and np.max(np.abs(wave)) > 0:
        idx0 = int(delay_d * rendering.sample_rate)
        idx1 = min(len(wave), idx0 + int(0.02 * rendering.sample_rate))
        peak = np.max(np.abs(wave[idx0:idx1])) if idx1 > idx0 else np.max(np.abs(wave))
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
