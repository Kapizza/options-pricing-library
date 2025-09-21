# tests/test_svi_surface.py
import math
import numpy as np
import pytest


from src.black_scholes import black_scholes_price
from src.svi_surface import (
        SVIParams,
        fit_svi_expiry_from_ivs,
        fit_svi_expiry_from_prices,
        fit_svi_surface,
        svi_total_variance,
    )

def _make_synthetic_chain_iv(S0, r, q, T, K):
    """
    Generate a convex, skewed smile via raw-SVI and return IVs at strikes.
    This is the ground-truth used in multiple tests.
    """
    F = S0 * math.exp((r - q) * T)
    k = np.log(K / F)
    # A mild, realistic SVI set
    true = SVIParams(a=0.015, b=0.75, rho=-0.45, m=0.0, sigma=0.22)
    w = svi_total_variance(k, true)
    iv = np.sqrt(np.maximum(w, 1e-12) / max(T, 1e-8))
    return iv, true


def _numeric_convex(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 5:
        return True
    h = np.gradient(x)
    wpp = (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / ((0.5 * (h + np.roll(h, 1))) ** 2 + 1e-16)
    wpp = wpp[1:-1]
    return np.all(wpp >= -1e-7)



def test_fit_svi_expiry_from_ivs_recovers_smile():
    np.random.seed(0)
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.5
    K = np.linspace(70, 130, 31)

    iv, _true = _make_synthetic_chain_iv(S0, r, q, T, K)
    # add small noise
    iv_noisy = np.clip(iv + 0.002 * np.random.randn(iv.size), 0.01, 5.0)
    F = S0 * math.exp((r - q) * T)

    p = fit_svi_expiry_from_ivs(K, iv_noisy, T, F)

    k = np.log(K / F)
    w_fit = svi_total_variance(k, p)
    w_true = (iv ** 2) * T
    rmse = np.sqrt(np.mean((w_fit - w_true) ** 2))
    assert rmse < 1.2e-3

    # Also check shape agreement with a scale-free metric (R^2 close to 1).
    ss_res = np.sum((w_fit - w_true) ** 2)
    ss_tot = np.sum((w_true - w_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    assert r2 > 0.995


@pytest.mark.parametrize("tenors", [[0.1, 0.25, 0.5, 1.0], [0.03, 0.08, 0.2, 0.4]])
def test_calendar_no_arb_and_nonnegative_variance(tenors):
    np.random.seed(1)
    S0, r, q = 100.0, 0.02, 0.0
    tenors = np.array(tenors, dtype=float)

    chains = {}
    for T in tenors:
        K = np.linspace(70, 130, 41)
        iv, _ = _make_synthetic_chain_iv(S0, r, q, T, K)
        iv = np.clip(iv + 0.003 * np.random.randn(iv.size), 0.01, 5.0)
        chains[T] = {"K": K, "iv": iv}

    surf = fit_svi_surface(chains, S0=S0, r=r, q=q, mode="iv")

    # Pick a few k locations and check calendar monotonicity of w/T
    for kval in [-0.4, 0.0, 0.3]:
        w_vals = []
        for T in tenors:
            w = (surf.iv(np.array([kval]), T).item() ** 2) * T
            w_vals.append(w)
        w_vals = np.array(w_vals)
        assert np.all(np.diff(w_vals) >= -1e-6)  # non-decreasing w w.r.t. T

    # Nonnegative variance and numeric convexity along a grid
    Kgrid = np.linspace(60, 140, 61)
    for T in tenors:
        F = S0 * math.exp((r - q) * T)
        kgrid = np.log(Kgrid / F)
        w = (surf.iv(kgrid, T) ** 2) * T
        assert np.all(w > 0.0)
        assert _numeric_convex(w, kgrid)


def test_fit_svi_from_prices_path_matches_iv_path():
    np.random.seed(2)
    S0, r, q = 100.0, 0.02, 0.0
    T = 0.4
    K = np.linspace(75, 125, 31)

    iv, _ = _make_synthetic_chain_iv(S0, r, q, T, K)

    # Build mid call prices at those IVs
    call_mid = np.array([
        black_scholes_price(S0, float(k), T, r, iv_i, option_type="call")
        for k, iv_i in zip(K, iv)
    ])

    # Fit using prices path
    p = fit_svi_expiry_from_prices(S0, r, q, T, K, call_mid)
    F = S0 * math.exp((r - q) * T)
    k = np.log(K / F)

    # Compare total variance shapes
    w_fit = svi_total_variance(k, p)
    w_true = (iv ** 2) * T
    mae = np.mean(np.abs(w_fit - w_true))
    assert mae < 3e-4


def test_surface_iv_consistency_roundtrip():
    np.random.seed(3)
    S0, r, q = 100.0, 0.01, 0.0
    tenors = np.array([0.05, 0.2, 0.7])
    chains = {}
    for T in tenors:
        K = np.linspace(80, 120, 25)
        iv, _ = _make_synthetic_chain_iv(S0, r, q, T, K)
        chains[T] = {"K": K, "iv": iv}

    surf = fit_svi_surface(chains, S0=S0, r=r, q=q, mode="iv")

    # Pick (k,T), compute w -> iv -> w again; should be stable
    for T in tenors:
        F = S0 * math.exp((r - q) * T)
        k = np.linspace(-0.3, 0.3, 21)
        iv1 = surf.iv(k, T)
        w1 = (iv1 ** 2) * T
        iv2 = np.sqrt(np.maximum(w1, 1e-12) / T)
        assert np.allclose(iv1, iv2, rtol=0, atol=1e-12)


def test_short_maturity_stability_and_monotonicity():
    np.random.seed(4)
    S0, r, q = 100.0, 0.015, 0.0
    tenors = np.array([0.02, 0.05, 0.1, 0.2])

    chains = {}
    for T in tenors:
        K = np.linspace(85, 115, 23)
        iv, _ = _make_synthetic_chain_iv(S0, r, q, T, K)
        # Slightly larger noise at very short maturities
        iv = np.clip(iv + 0.004 * np.random.randn(iv.size), 0.01, 5.0)
        chains[T] = {"K": K, "iv": iv}

    surf = fit_svi_surface(chains, S0=S0, r=r, q=q, mode="iv")

    # Check calendar monotonicity at ATM-ish k=0
    k0 = np.array([0.0])
    vals = np.array([(surf.iv(k0, T)[0] ** 2) * T for T in tenors])
    assert np.all(np.diff(vals) >= -1e-6)
