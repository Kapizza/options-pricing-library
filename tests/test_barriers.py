import math
import numpy as np
import pytest

from src.barriers import (
    barrier_price_mc,
    MCConfig,
    digital_cash_bsm,
    digital_asset_bsm,
)
from src.black_scholes import black_scholes_price


def approx(a, b, rel=5e-2, abs_=5e-4):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("barrier", ["up-and-out", "down-and-out"])
def test_in_out_parity(option, barrier):
    S0, K, T, r, q, sigma = 100.0, 100.0, 0.75, 0.01, 0.0, 0.2
    H = 120.0 if barrier.startswith("up") else 80.0

    cfg = MCConfig(n_paths=200_000, n_steps=365, seed=42, antithetic=True)

    price_out = barrier_price_mc(
        S0, K, H, T, r, q, sigma, option=option, barrier=barrier, cfg=cfg
    )

    # Use the same MC engine to price knock in via parity in barriers.py
    price_in = barrier_price_mc(
        S0,
        K,
        H,
        T,
        r,
        q,
        sigma,
        option=option,
        barrier=("up-and-in" if barrier == "up-and-out" else "down-and-in"),
        cfg=cfg,
    )

    vanilla = black_scholes_price(S0, K, T, r, sigma, option_type=option, q=q)

    assert approx(price_in + price_out, vanilla, rel=1e-2, abs_=2e-3)


def test_up_and_out_zero_when_spot_above_barrier():
    # If S0 >= H for up-and-out, the option is immediately knocked out
    S0, K, H = 105.0, 100.0, 100.0
    T, r, q, sigma = 1.0, 0.0, 0.0, 0.2
    cfg = MCConfig(n_paths=50_000, n_steps=250, seed=7)
    price = barrier_price_mc(S0, K, H, T, r, q, sigma, option="call", barrier="up-and-out", cfg=cfg)
    assert price < 1e-3


@pytest.mark.parametrize("barrier", ["up-and-out", "down-and-out"])
def test_rebate_increases_knock_out_price(barrier):
    S0, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.01, 0.0, 0.25
    H = 120.0 if barrier.startswith("up") else 80.0
    cfg = MCConfig(n_paths=100_000, n_steps=252, seed=123)

    p0 = barrier_price_mc(S0, K, H, T, r, q, sigma, option="call", barrier=barrier, rebate=0.0, cfg=cfg)
    p1 = barrier_price_mc(S0, K, H, T, r, q, sigma, option="call", barrier=barrier, rebate=5.0, cfg=cfg)

    assert p1 > p0


def test_time_discretization_convergence():
    # Price should be stable as we increase time steps
    S0, K, H = 100.0, 100.0, 120.0
    T, r, q, sigma = 1.0, 0.01, 0.0, 0.2
    cfg1 = MCConfig(n_paths=150_000, n_steps=126, seed=99)
    cfg2 = MCConfig(n_paths=150_000, n_steps=252, seed=99)
    cfg3 = MCConfig(n_paths=150_000, n_steps=504, seed=99)

    p1 = barrier_price_mc(S0, K, H, T, r, q, sigma, option="call", barrier="up-and-out", cfg=cfg1)
    p2 = barrier_price_mc(S0, K, H, T, r, q, sigma, option="call", barrier="up-and-out", cfg=cfg2)
    p3 = barrier_price_mc(S0, K, H, T, r, q, sigma, option="call", barrier="up-and-out", cfg=cfg3)

    # successive differences should be small
    assert abs(p2 - p1) < 0.05
    assert abs(p3 - p2) < 0.04


@pytest.mark.parametrize("option", ["call", "put"])
def test_digitals_bounds_and_monotonicity(option):
    # Basic sanity checks for digitals
    S0, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.02, 0.01, 0.3

    cash = digital_cash_bsm(S0, K, T, r, sigma, q=q, option=option, cash=1.0)
    asset = digital_asset_bsm(S0, K, T, r, sigma, q=q, option=option)

    # Bounds
    assert 0.0 <= cash <= math.exp(-r * T) + 1e-12
    assert 0.0 <= asset <= S0 * math.exp(-q * T) + 1e-9

    # Monotonicity in strike: for calls, cash digital decreases with K; for puts, increases
    cash_K_up = digital_cash_bsm(S0, K + 1.0, T, r, sigma, q=q, option=option, cash=1.0)
    if option == "call":
        assert cash_K_up <= cash + 1e-12
    else:
        assert cash_K_up >= cash - 1e-12



def test_in_out_parity_down_barrier_put_strict():
    # A second strict parity test on a more extreme set of params
    S0, K, T, r, q, sigma = 90.0, 100.0, 1.25, 0.03, 0.01, 0.35
    H = 70.0
    cfg = MCConfig(n_paths=300_000, n_steps=365, seed=2024)

    p_out = barrier_price_mc(S0, K, H, T, r, q, sigma, option="put", barrier="down-and-out", cfg=cfg)
    p_in = barrier_price_mc(S0, K, H, T, r, q, sigma, option="put", barrier="down-and-in", cfg=cfg)
    vanilla = black_scholes_price(S0, K, T, r, sigma, q=q, option_type="put")

    # Tight tolerance with many paths
    assert approx(p_in + p_out, vanilla, rel=6e-3, abs_=2e-3)
