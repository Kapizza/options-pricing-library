import math
import numpy as np
import pytest

from src.jumps import (
    cf_merton,
    merton_price_cos,
    merton_call_put_parity,
)
from src.black_scholes import black_scholes_price


def _close(a, b, rel=2e-3, abs_=2e-4, imag_tol=1e-12):
    a = complex(a)
    return (
        math.isclose(a.real, float(b), rel_tol=rel, abs_tol=abs_) and
        abs(a.imag) <= imag_tol
    )


@pytest.mark.parametrize("option", ["call", "put"])
def test_merton_reduces_to_bs_when_no_jumps(option):
    # When lam = 0, Merton JD collapses to plain Black–Scholes
    S0, K, T = 100.0, 95.0, 0.8
    r, q, sigma = 0.02, 0.01, 0.25

    lam, muJ, sigJ = 0.0, 0.0, 0.2

    px_merton = merton_price_cos(S0, K, T, r, q, sigma, lam, muJ, sigJ, option=option, N=2048, L=12)
    px_bs = black_scholes_price(S0, K, T, r, sigma, option_type=option, q=q)

    assert _close(px_merton, px_bs, rel=2e-3, abs_=3e-4)


def test_call_put_parity_holds():
    S0, K, T = 100.0, 100.0, 1.0
    r, q, sigma = 0.01, 0.00, 0.20

    lam, muJ, sigJ = 0.5, 0.0, 0.25

    c = merton_price_cos(S0, K, T, r, q, sigma, lam, muJ, sigJ, option="call", N=2048, L=12)
    p = merton_price_cos(S0, K, T, r, q, sigma, lam, muJ, sigJ, option="put",  N=2048, L=12)

    p_from_parity = merton_call_put_parity(S0, K, T, r, q, c)

    assert _close(p, p_from_parity, rel=2e-3, abs_=3e-4)


def test_option_price_increases_with_jump_intensity_atm_call():
    # At-the-money call typically increases with added jump variance (muJ=0)
    S0, K, T = 100.0, 100.0, 0.5
    r, q, sigma = 0.01, 0.00, 0.20
    muJ, sigJ = 0.0, 0.25

    c0  = merton_price_cos(S0, K, T, r, q, sigma, 0.0, muJ, sigJ, option="call", N=2048, L=12)
    c05 = merton_price_cos(S0, K, T, r, q, sigma, 0.5, muJ, sigJ, option="call", N=2048, L=12)
    c10 = merton_price_cos(S0, K, T, r, q, sigma, 1.0, muJ, sigJ, option="call", N=2048, L=12)

    assert c05 > c0 - 1e-8
    assert c10 > c05 - 1e-8


def test_vectorized_strikes_and_monotonicity_call():
    S0, T = 100.0, 0.75
    r, q, sigma = 0.01, 0.00, 0.20
    lam, muJ, sigJ = 0.4, 0.0, 0.2

    Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = merton_price_cos(S0, Ks, T, r, q, sigma, lam, muJ, sigJ, option="call", N=2048, L=12)

    assert isinstance(prices, np.ndarray)
    assert prices.shape == Ks.shape

    # Call prices should be non-increasing in K (weakly, due to numerical noise)
    assert np.all(np.diff(prices) <= 1e-8)


@pytest.mark.parametrize("muJ, sigJ", [(0.0, 0.2), (-0.1, 0.3)])
def test_cf_merton_unit_value_at_zero(muJ, sigJ):
    # CF at u=0 should be 1 (normalized characteristic function)
    T, r, q, sigma, lam = 0.7, 0.01, 0.0, 0.2, 0.5
    val = cf_merton(0.0, T, r, q, sigma, lam, muJ, sigJ)
    assert _close(val, 1.0, rel=0.0, abs_=1e-12)


# ---------------------------------------------------------------------------
# Regression: strike-centred window and bounded (put) payoff in merton_price_cos
# ---------------------------------------------------------------------------
import itertools


def _bs(S0, K, T, r, vol, q, option):
    # Own Black-Scholes (normal CDF from erfc) for the reference below.
    N = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    F = S0 * math.exp((r - q) * T)
    s = vol * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    if option == "call":
        return math.exp(-r * T) * (F * N(d1) - K * N(d2))
    return math.exp(-r * T) * (K * N(-d2) - F * N(-d1))


def _merton_series(S0, K, T, r, q, sigma, lam, muJ, sigJ, option):
    """Merton (1976) Poisson series of Black-Scholes prices:
    sum_n e^{-lam' T} (lam' T)^n / n! BS(S0, K, T, r_n, sigma_n, q), with
    kJ = e^{muJ + sigJ^2/2} - 1, lam' = lam (1 + kJ), sigma_n^2 = sigma^2 + n sigJ^2/T,
    r_n = r - lam kJ + n ln(1 + kJ)/T  (conditioning cf_merton on n jumps)."""
    kJ = math.exp(muJ + 0.5 * sigJ * sigJ) - 1.0
    m = lam * (1.0 + kJ) * T
    if m == 0.0:
        return _bs(S0, K, T, r, sigma, q, option)
    total = 0.0
    for n in range(int(m + 15.0 * math.sqrt(m) + 50) + 1):
        w = math.exp(-m + n * math.log(m) - math.lgamma(n + 1.0))
        sig_n = math.sqrt(sigma * sigma + n * sigJ * sigJ / T)
        r_n = r - lam * kJ + n * math.log1p(kJ) / T
        total += w * _bs(S0, K, T, r_n, sig_n, q, option)
    return total


def test_merton_series_reference_reduces_to_black_scholes():
    for option in ("call", "put"):
        ref = black_scholes_price(100.0, 95.0, 0.8, 0.02, 0.25, option_type=option, q=0.01)
        assert abs(_merton_series(100.0, 95.0, 0.8, 0.02, 0.01, 0.25, 0.0, -0.1, 0.2, option) - ref) < 1e-13


@pytest.mark.parametrize("args, call_ref", [
    ((100.0, 100.0, 20.0, 0.03, 0.01, 0.8, 5.0, -0.3, 0.6), 81.8339731973),
    ((100.0, 60.0, 0.25, 0.03, 0.01, 0.15, 0.1, 0.05, 0.3), 40.2044543075),
])
def test_merton_cos_matches_poisson_series_reported_cases(args, call_ref):
    # Before: 2.41e14 (call payoff e^b blow-up on a wide window), and 40.1459,
    # below the no-arbitrage lower bound 40.1986 (window not centred on K = 60).
    S0, K, T, r, q, sigma, lam, muJ, sigJ = args
    assert abs(_merton_series(*args, "call") - call_ref) < 1e-9
    lower = max(S0 * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    for option in ("call", "put"):
        ref = _merton_series(*args, option)
        px = merton_price_cos(*args, option=option)
        assert abs(px - ref) <= 1e-6 * ref + 1e-8
    assert lower <= merton_price_cos(*args, option="call") <= S0 * math.exp(-q * T)


def test_merton_cos_matches_poisson_series_on_grid():
    S0, r, q = 100.0, 0.03, 0.01
    Ks = np.array([60.0, 80.0, 100.0, 125.0, 160.0])
    laws = [(-0.1, 0.15), (0.05, 0.3), (-0.3, 0.6), (0.2, 0.1)]
    for T, sigma, lam, (muJ, sigJ) in itertools.product([0.25, 1.0, 5.0], [0.1, 0.2, 0.4], [0.1, 0.5, 1.0], laws):
        for option in ("call", "put"):
            px = merton_price_cos(S0, Ks, T, r, q, sigma, lam, muJ, sigJ, option=option)
            ref = np.array([_merton_series(S0, K, T, r, q, sigma, lam, muJ, sigJ, option) for K in Ks])
            assert np.all(np.abs(px - ref) <= 1e-6 * ref + 1e-8), (T, sigma, lam, muJ, sigJ, option, px - ref)


def test_merton_cos_return_components_per_strike():
    Ks = np.array([80.0, 100.0, 125.0])
    px, (u, a, b, F) = merton_price_cos(100.0, Ks, 1.0, 0.02, 0.0, 0.2, 0.5, -0.1, 0.15, return_components=True)
    assert np.allclose(px, merton_price_cos(100.0, Ks, 1.0, 0.02, 0.0, 0.2, 0.5, -0.1, 0.15))
    assert a.shape == b.shape == (3,) and F.shape == (3, u.size)
    # windows are centred per strike on the mean of ln(S_T/K) and share one width
    assert np.allclose(b - a, b[0] - a[0])
    assert np.allclose(0.5 * (a + b) + np.log(Ks), 0.5 * (a[0] + b[0]) + np.log(Ks[0]))
