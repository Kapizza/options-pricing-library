# tests/test_hurst.py
import numpy as np
import pytest
from src.hurst import hurst_rs, hurst_dfa


def test_white_noise_has_hurst_near_half():
    rng = np.random.RandomState(123)
    x = rng.randn(5000)
    H_rs = hurst_rs(x)
    H_dfa = hurst_dfa(x, order=1)
    assert 0.4 < H_rs < 0.6, f"R/S H={H_rs}"
    assert 0.4 < H_dfa < 0.6, f"DFA H={H_dfa}"


def test_persistent_series_has_hurst_gt_half():
    rng = np.random.RandomState(123)
    x = np.cumsum(rng.randn(5000))  # integrated white noise ~ trending
    H_rs = hurst_rs(x)
    H_dfa = hurst_dfa(x, order=1)
    assert H_rs > 0.6, f"R/S H={H_rs}"
    assert H_dfa > 0.6, f"DFA H={H_dfa}"


def test_anti_persistent_series_has_hurst_lt_half():
    rng = np.random.RandomState(123)
    n = 10000
    # AR(1) with strong negative autocorrelation (anti-persistent)
    phi = -0.8
    eps = rng.randn(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t-1] + eps[t]

    H_rs  = hurst_rs(x)
    H_dfa = hurst_dfa(x, order=1)

    # R/S has positive bias; use a slightly looser threshold
    assert H_rs  < 0.5,  f"R/S H={H_rs}"
    assert H_dfa < 0.45, f"DFA H={H_dfa}"



@pytest.mark.parametrize("method", [hurst_rs, hurst_dfa])
def test_short_series_returns_nan(method):
    x = np.arange(10)  # too short
    H = method(x)
    assert np.isnan(H)
