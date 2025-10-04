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
