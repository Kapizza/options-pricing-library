# tests/test_svi_surface.py
import math
import warnings
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
    # A mild, realistic SVI set. b was 0.75, which has butterfly arbitrage:
    # g(k) < 0 for k in [-1.31, -0.38], and the Breeden-Litzenberger density
    # from these IVs is negative for K in [27, 69] at T=0.5 (F=101). An
    # arbitrage-free fitter cannot reproduce it (best arbitrage-free rmse 2.3e-3).
    # b=0.6 is the nearest arbitrage-free slice with a, rho, m, sigma unchanged.
    true = SVIParams(a=0.015, b=0.6, rho=-0.45, m=0.0, sigma=0.22)
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


def test_fit_from_prices_with_dividend_yield():
    # Call prices generated with q > 0 must be inverted with q (not with r - q),
    # otherwise the fitted IVs are biased (3+ vol points here).
    S0, r, q, T = 100.0, 0.03, 0.04, 0.5
    F = S0 * math.exp((r - q) * T)
    k = np.linspace(-0.3, 0.3, 25)
    K = F * np.exp(k)
    true = SVIParams(a=0.015, b=0.3, rho=-0.4, m=0.0, sigma=0.15)
    iv_true = np.sqrt(svi_total_variance(k, true) / T)
    calls = np.array([black_scholes_price(S0, Ki, T, r, s, option_type="call", q=q)
                      for Ki, s in zip(K, iv_true)])
    p = fit_svi_expiry_from_prices(S0, r, q, T, K, calls)
    iv_fit = np.sqrt(svi_total_variance(k, p) / T)
    assert np.max(np.abs(iv_fit - iv_true)) < 2e-3


# ---------------------------------------------------------------------------
# Regression: butterfly (density) no-arbitrage, Gatheral & Jacquier (2014)
# g(k) = (1 - k w'/(2w))^2 - w'^2/4 (1/w + 1/4) + w''/2 >= 0
# ---------------------------------------------------------------------------
from src.svi_surface import SVISurface, svi_butterfly_g

# Gatheral & Jacquier (2014), Example 3.1 (Axel Vogt): a raw SVI slice with
# butterfly arbitrage, T = 1.
_VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_butterfly_g_flags_known_arbitrage():
    k = np.linspace(-1.5, 1.5, 601)
    assert svi_butterfly_g(k, _VOGT).min() < -0.02
    good = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.2)
    assert svi_butterfly_g(k, good).min() > 0.0


def test_butterfly_g_flags_nonpositive_variance():
    # w(k) = -0.05 + 0.5 sqrt(k^2 + 0.05^2) is negative for |k| < 0.0866.
    # Negative total variance is itself an arbitrage, so g must flag it; w was
    # clamped to 1e-300, which gave g = +inf, 6.0, +inf at k = -0.05, 0, 0.05.
    p = SVIParams(a=-0.05, b=0.5, rho=0.0, m=0.0, sigma=0.05)
    k = np.array([-0.2, -0.05, 0.0, 0.05, 0.2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = svi_butterfly_g(k, p)
    assert np.all(svi_total_variance(k[1:4], p) < 0.0)
    assert np.all(np.isneginf(g[1:4]))
    # where w > 0, g is unchanged: compare with finite differences of w
    h = 1e-4
    kk = k[[0, 4]]
    w0 = svi_total_variance(kk, p)
    wp, wm = svi_total_variance(kk + h, p), svi_total_variance(kk - h, p)
    w1, w2 = (wp - wm) / (2 * h), (wp - 2 * w0 + wm) / h ** 2
    g_fd = (1 - kk * w1 / (2 * w0)) ** 2 - w1 ** 2 / 4 * (1 / w0 + 0.25) + w2 / 2
    assert np.allclose(g[[0, 4]], g_fd, rtol=1e-6)
    assert np.allclose(g[[0, 4]], -1.0442114, atol=1e-6)
    assert not [c for c in caught if issubclass(c.category, RuntimeWarning)]

    # the stitched-surface version: rows are clipped at w = 0 by the convexity
    # repair, and those points must be flagged too, not reported as g = 1 or inf
    kg = np.linspace(-0.5, 0.5, 101)
    w_row = np.maximum(svi_total_variance(kg, p), 0.0)
    surf = SVISurface(tenors=np.array([1.0]), params=[p], k_grid=kg, w_grid=w_row[None, :])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gs = surf.butterfly_g()[0]
    assert np.all(np.isneginf(gs[w_row <= 0.0]))
    assert np.all(np.isfinite(gs[w_row > 0.0]))
    assert not [c for c in caught if issubclass(c.category, RuntimeWarning)]


@pytest.mark.parametrize("case", ["vogt", "repo_smile"])
def test_fitted_slices_are_butterfly_free(case):
    if case == "vogt":
        T, F = 1.0, 100.0
        k = np.linspace(-1.5, 1.5, 61)
        iv = np.sqrt(svi_total_variance(k, _VOGT) / T)
    else:   # a smile with arbitrage just outside the quoted strikes (k < -0.38)
        T, F = 0.5, 100.0 * math.exp(0.02 * 0.5)
        k = np.log(np.linspace(70, 130, 31) / F)
        bad = SVIParams(a=0.015, b=0.75, rho=-0.45, m=0.0, sigma=0.22)
        iv = np.sqrt(svi_total_variance(k, bad) / T)
    p = fit_svi_expiry_from_ivs(F * np.exp(k), iv, T, F)
    kk = np.linspace(k.min() - 1.0, k.max() + 1.0, 801)
    assert svi_butterfly_g(kk, p).min() >= -1e-6
    assert p.b * (1.0 + abs(p.rho)) <= 2.0 + 1e-9            # Roger Lee wing bound
    # still close where there is data (best arbitrage-free rmse for the
    # repo_smile case is 2.3e-3 by an independent SLSQP fit)
    w_fit = svi_total_variance(k, p)
    assert np.sqrt(np.mean((w_fit - iv ** 2 * T) ** 2)) < 3e-3


# Review cases (F = 100). Cases 1 and 2 are arbitrage-free SSVI smiles plus
# 0.3 vol-pt noise; an unchecked, failed SLSQP polish turned them into a
# collapsed slice (case 1: w ~ 0, IV 0 at every strike) and IVs up to 0.26 off
# (case 2). Case 3 (one day, sigma ~ 0.01) had g pinned at 1e-4 on a fixed grid
# of spacing 0.0034 but g = -4.4e-4 between its points, with no warning. Its
# quotes carry static arbitrage themselves (the call price rises with strike
# from K = 100.84 to 101.58), so no arbitrage-free smile fits them to better
# than 1.2 vol pts RMSE, and the best arbitrage-free raw SVI fit (least
# squares in w) is 8.3 vol pts RMSE: its bounds are set from that.
_REVIEW_SMILES = {
    "case1": (3 / 252,
              [86.386611, 88.047417, 89.740153, 91.465432, 93.22388, 95.016135, 96.842847,
               98.704677, 100.602302, 102.536409, 104.5077, 106.516889, 108.564706,
               110.651892, 112.779205],
              [0.623678, 0.597806, 0.576369, 0.547432, 0.516571, 0.485084, 0.457563,
               0.421932, 0.388889, 0.342584, 0.310225, 0.268858, 0.232775, 0.198307,
               0.187125],
              0.01, 0.01),
    "case2": (0.5,
              [75.497457, 80.542965, 85.925665, 91.668093, 97.794287, 104.329896,
               111.302281, 118.740632, 126.676089],
              [0.307675, 0.284705, 0.259868, 0.232425, 0.202771, 0.168611, 0.133608,
               0.097574, 0.090564],
              0.01, 0.01),
    "case3": (0.004,
              [93.745653, 94.432176, 95.123726, 95.820341, 96.522058, 97.228913,
               97.940945, 98.658191, 99.38069, 100.108479, 100.841599, 101.580087,
               102.323984, 103.073328, 103.82816],
              [0.875029, 0.814427, 0.738295, 0.664709, 0.595783, 0.503632, 0.406872,
               0.287646, 0.216668, 0.286087, 0.399098, 0.555661, 0.698607, 0.822628,
               0.940389],
              0.085, 0.175),
}


def _butterfly_status_fine(p, k):
    """min g on >= 200k points over the quoted range +- 2, log-spaced wings out
    to |k| = 50 and a fine grid around the vertex; b (1 + |rho|); min w."""
    lo, hi = k.min() - 2.0, k.max() + 2.0
    kk = np.concatenate([np.linspace(lo, hi, 200001),
                         -np.geomspace(-lo, 50.0, 2000), np.geomspace(hi, 50.0, 2000),
                         p.m + p.sigma * np.linspace(-50.0, 50.0, 20001)])
    w_min = p.a + p.b * p.sigma * math.sqrt(1.0 - p.rho ** 2)
    return svi_butterfly_g(kk, p).min(), p.b * (1.0 + abs(p.rho)), w_min


@pytest.mark.parametrize("case", ["case1", "case2", "case3"])
def test_review_smiles_fit_and_are_butterfly_free(case):
    T, K, iv, rmse_max, err_max = _REVIEW_SMILES[case]
    K, iv, F = np.array(K), np.array(iv), 100.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = fit_svi_expiry_from_ivs(K, iv, T, F)
    k = np.log(K / F)
    err = np.sqrt(np.maximum(svi_total_variance(k, p), 0.0) / T) - iv
    assert np.sqrt(np.mean(err ** 2)) < rmse_max
    assert np.max(np.abs(err)) < err_max
    g_min, lee, w_min = _butterfly_status_fine(p, k)
    assert g_min >= -1e-6
    assert lee <= 2.0
    assert w_min > 0.0
    assert not [c for c in caught if issubclass(c.category, RuntimeWarning)]


def test_surface_butterfly_check():
    np.random.seed(5)
    S0, r, q = 100.0, 0.02, 0.0
    chains = {}
    for T in (0.1, 0.25, 0.5, 1.0):
        K = np.linspace(70, 130, 41)
        iv, _ = _make_synthetic_chain_iv(S0, r, q, T, K)
        chains[T] = {"K": K, "iv": iv}
    surf = fit_svi_surface(chains, S0=S0, r=r, q=q, mode="iv")
    g = surf.butterfly_g()
    assert g.shape == surf.w_grid.shape
    assert np.nanmin(g[:, 2:-2]) >= -1e-3


def test_iv_is_flat_outside_quoted_tenors():
    # Below the first tenor, total variance must go to 0 with T (flat IV);
    # linear extrapolation in w made the ATM IV explode as T -> 0.
    S0, r, q = 100.0, 0.02, 0.0
    truths = {0.05: SVIParams(0.002, 0.10, -0.6, 0.01, 0.08),
              0.25: SVIParams(0.008, 0.12, -0.5, 0.02, 0.12),
              0.5: SVIParams(0.016, 0.13, -0.45, 0.02, 0.15),
              1.0: SVIParams(0.03, 0.14, -0.4, 0.03, 0.2)}
    chains = {}
    for T, p in truths.items():
        F = S0 * math.exp((r - q) * T)
        kk = np.linspace(-0.5, 0.35, 35)
        chains[T] = {"K": F * np.exp(kk), "iv": np.sqrt(svi_total_variance(kk, p) / T)}
    surf = fit_svi_surface(chains, S0=S0, r=r, q=q, mode="iv")
    k = np.array([-0.2, 0.0, 0.2])
    iv_first, iv_last = surf.iv(k, 0.05), surf.iv(k, 1.0)
    for T in (0.03, 0.01, 0.002):
        assert np.allclose(surf.iv(k, T), iv_first, rtol=1e-10)
    for T in (1.5, 3.0):
        assert np.allclose(surf.iv(k, T), iv_last, rtol=1e-10)
    # interpolation between tenors is unchanged: w at a quoted tenor is the grid row
    assert np.allclose(surf.w(surf.k_grid, 0.25), surf.w_grid[list(surf.tenors).index(0.25)])
