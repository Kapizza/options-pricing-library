import numpy as np
import pytest

from src.black_scholes import (
    black_scholes_price,
    black_scholes_price_from_forward,
    bs_call_forward,
    bs_put_forward,
    bs_d1_d2,
    implied_vol_from_price,
)

# -----------------------------
# 1) Basic prices (q=0, q>0)
# -----------------------------

def test_prices_q0_and_qpos():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # q = 0
    c0 = black_scholes_price(S, K, T, r, sigma, "call", q=0.0)
    p0 = black_scholes_price(S, K, T, r, sigma, "put",  q=0.0)
    # smoke + reasonable ranges
    assert 10.0 < c0 < 12.0
    assert 5.0  < p0 < 7.0

    # q > 0 lowers calls, raises puts (all else equal)
    q = 0.02
    c1 = black_scholes_price(S, K, T, r, sigma, "call", q=q)
    p1 = black_scholes_price(S, K, T, r, sigma, "put",  q=q)
    assert c1 < c0
    assert p1 > p0


# -----------------------------
# 2) Put–call parity (with q)
# -----------------------------

def test_put_call_parity_with_q():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.22, 0.03
    c = black_scholes_price(S, K, T, r, sigma, "call", q=q)
    p = black_scholes_price(S, K, T, r, sigma, "put",  q=q)
    lhs = c - p
    rhs = S * np.exp(-q*T) - K * np.exp(-r*T)
    assert np.isclose(lhs, rhs, atol=1e-10, rtol=1e-10)


# -----------------------------
# 3) Vectorization
# -----------------------------

def test_vectorized_bs_d1_d2_and_price():
    S = np.array([90.0, 100.0, 110.0])
    K, T, r, sigma, q = 100.0, 0.5, 0.02, 0.25, 0.01

    d = bs_d1_d2(S, K, T, r, sigma, q=q)
    assert "d1" in d and "d2" in d
    assert d["d1"].shape == S.shape
    assert d["d2"].shape == S.shape

    # price vectorization
    calls = black_scholes_price(S, K, T, r, sigma, "call", q=q)
    puts  = black_scholes_price(S, K, T, r, sigma, "put",  q=q)
    assert isinstance(calls, np.ndarray) and calls.shape == S.shape
    assert isinstance(puts,  np.ndarray) and puts.shape  == S.shape
    assert np.all(calls > 0) and np.all(puts > 0)


# -----------------------------
# 4) Forward–measure equivalence
# -----------------------------

def test_forward_measure_equivalence():
    S, K, T, r, sigma, q = 120.0, 110.0, 0.75, 0.03, 0.18, 0.015
    disc_r = np.exp(-r*T)
    disc_q = np.exp(-q*T)
    F = S * disc_q / disc_r  # forward under q

    # spot measure price
    c_spot = black_scholes_price(S, K, T, r, sigma, "call", q=q)
    p_spot = black_scholes_price(S, K, T, r, sigma, "put",  q=q)

    # forward measure (undiscounted), then discount with DF
    c_fwd = black_scholes_price_from_forward(F, disc_r, K, T, sigma, option_type="call")
    p_fwd = black_scholes_price_from_forward(F, disc_r, K, T, sigma, option_type="put")

    assert np.isclose(c_spot, c_fwd, rtol=1e-12, atol=1e-12)
    assert np.isclose(p_spot, p_fwd, rtol=1e-12, atol=1e-12)

    # direct forward functions also agree (undiscounted)
    c_und = bs_call_forward(F, K, T, sigma)
    p_und = bs_put_forward(F, K, T, sigma)
    assert np.isclose(c_fwd, disc_r * c_und, rtol=1e-12, atol=1e-12)
    assert np.isclose(p_fwd, disc_r * p_und, rtol=1e-12, atol=1e-12)


# -----------------------------
# 5) Implied vol round-trip (with q)
# -----------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_roundtrip_with_q(option_type):
    S, K, T, r, sigma_true, q = 100.0, 95.0, 0.8, 0.01, 0.33, 0.02
    price = black_scholes_price(S, K, T, r, sigma_true, option_type=option_type, q=q)
    iv = implied_vol_from_price(S, K, T, r, price, option_type=option_type, q=q)
    assert np.isfinite(iv)
    assert np.isclose(iv, sigma_true, rtol=2e-4, atol=2e-6)
