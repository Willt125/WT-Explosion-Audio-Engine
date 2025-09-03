"""Tests for nonlinear Burgers propagation."""

import numpy as np

from ..explosion import AtmosphereWT, Geometry, Ordnance, Rendering, synthesize_explosion

def _rise_time(w, sr):
    idx = int(np.argmax(w))
    peak = w[idx]
    th = 0.1 * peak
    pre = w[:idx]
    start = np.where(pre >= th)[0][0]
    return (idx - start) / sr


def _max_slope(w, sr):
    return np.max(np.abs(np.diff(w))) * sr


def _time_to_neg(w, sr):
    idx = int(np.argmax(w))
    neg_idx = idx + np.argmin(w[idx: idx + int(0.05 * sr)])
    return (neg_idx - idx) / sr


def _render(distance, sr, use_burgers):
    ord = Ordnance(10.0)
    geo = Geometry(source=(0.0, 0.0, 1.0), receiver=(distance, 0.0, 1.7))
    atmos = AtmosphereWT(rh=0.5)
    render = Rendering(sample_rate=sr, pad=0.1, use_burgers=use_burgers, enable_source_shaper=False)
    return synthesize_explosion(ord, geo, atmos, render)


def test_burgers_50m():
    sr = 16000
    w_lin = _render(50.0, sr, False)
    w_nl = _render(50.0, sr, True)
    assert _max_slope(w_nl, sr) > _max_slope(w_lin, sr)
    assert _time_to_neg(w_nl, sr) < _time_to_neg(w_lin, sr)


def test_burgers_200m():
    sr = 16000
    w_lin = _render(200.0, sr, False)
    w_nl = _render(200.0, sr, True)
    assert _rise_time(w_nl, sr) < _rise_time(w_lin, sr)
    assert _time_to_neg(w_nl, sr) < _time_to_neg(w_lin, sr)
