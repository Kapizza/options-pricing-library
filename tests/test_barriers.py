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


# ---------------------------------------------------------------------------
# Regression: MC prices against closed-form continuous-barrier prices
# (Reiner & Rubinstein 1991; formulas as in Haug 2007, sec. 4.17.1).
# Knock-out rebates are paid at the hit time, knock-in rebates at expiry
# if the barrier is never hit.
# ---------------------------------------------------------------------------

def _haug_barrier(S, X, H, T, r, q, s, barrier, option, rebate=0.0):
    from scipy.stats import norm
    N = norm.cdf
    b = r - q
    sT = s * math.sqrt(T)
    mu = (b - 0.5 * s * s) / (s * s)
    lam = math.sqrt(mu * mu + 2.0 * r / (s * s))
    x1 = math.log(S / X) / sT + (1 + mu) * sT
    x2 = math.log(S / H) / sT + (1 + mu) * sT
    y1 = math.log(H * H / (S * X)) / sT + (1 + mu) * sT
    y2 = math.log(H / S) / sT + (1 + mu) * sT
    z = math.log(H / S) / sT + lam * sT
    eta = 1.0 if barrier.startswith("down") else -1.0
    phi = 1.0 if option == "call" else -1.0
    fq, fr = math.exp((b - r) * T), math.exp(-r * T)
    A = phi * S * fq * N(phi * x1) - phi * X * fr * N(phi * x1 - phi * sT)
    B = phi * S * fq * N(phi * x2) - phi * X * fr * N(phi * x2 - phi * sT)
    C = (phi * S * fq * (H / S) ** (2 * (mu + 1)) * N(eta * y1)
         - phi * X * fr * (H / S) ** (2 * mu) * N(eta * y1 - eta * sT))
    D = (phi * S * fq * (H / S) ** (2 * (mu + 1)) * N(eta * y2)
         - phi * X * fr * (H / S) ** (2 * mu) * N(eta * y2 - eta * sT))
    E = rebate * fr * (N(eta * x2 - eta * sT) - (H / S) ** (2 * mu) * N(eta * y2 - eta * sT))
    F = rebate * ((H / S) ** (mu + lam) * N(eta * z)
                  + (H / S) ** (mu - lam) * N(eta * z - 2 * eta * lam * sT))
    above = X > H
    table = {
        ("down-and-in", "call"): C + E if above else A - B + D + E,
        ("up-and-in", "call"): A + E if above else B - C + D + E,
        ("down-and-in", "put"): B - C + D + E if above else A + E,
        ("up-and-in", "put"): A - B + D + E if above else C + E,
        ("down-and-out", "call"): A - C + F if above else B - D + F,
        ("up-and-out", "call"): F if above else A - B + C - D + F,
        ("down-and-out", "put"): A - B + C - D + F if above else F,
        ("up-and-out", "put"): B - D + F if above else A - C + F,
    }
    return float(table[(barrier, option)])


# Haug (2007), Table 4-13: S=100, T=0.5, r=0.08, b=0.04, sigma=0.25, rebate=3
_HAUG_TABLE = [
    ("down-and-out", "call", 95, (9.0246, 6.7924, 4.8759)),
    ("up-and-out", "call", 105, (2.6789, 2.3580, 2.3453)),
    ("down-and-in", "call", 95, (7.7627, 4.0109, 2.0576)),
    ("up-and-in", "call", 105, (14.1112, 8.4482, 4.5910)),
    ("down-and-out", "put", 95, (2.2798, 2.2947, 2.6252)),
    ("up-and-out", "put", 105, (3.7760, 5.4932, 7.5187)),
    ("down-and-in", "put", 95, (2.9586, 6.5677, 11.9752)),
    ("up-and-in", "put", 105, (1.4653, 3.3721, 7.0846)),
]


@pytest.mark.parametrize("barrier, option, H, refs", _HAUG_TABLE)
def test_closed_form_reference_matches_haug_table(barrier, option, H, refs):
    for X, ref in zip((90.0, 100.0, 110.0), refs):
        val = _haug_barrier(100.0, X, H, 0.5, 0.08, 0.04, 0.25, barrier, option, rebate=3.0)
        assert abs(val - ref) < 6e-5


_MC_CASES = [
    ("up-and-out", "call", 100.0, 120.0),
    ("down-and-out", "call", 100.0, 90.0),
    ("down-and-out", "put", 100.0, 80.0),
    ("up-and-out", "put", 100.0, 110.0),
    ("up-and-in", "call", 100.0, 120.0),
    ("down-and-in", "call", 100.0, 90.0),
    ("down-and-in", "put", 100.0, 80.0),
    ("up-and-in", "put", 100.0, 110.0),
]


@pytest.mark.parametrize("barrier, option, K, H", _MC_CASES)
def test_mc_matches_closed_form_no_rebate(barrier, option, K, H):
    S0, T, r, q, sigma = 100.0, 1.0, 0.03, 0.01, 0.2
    cfg = MCConfig(n_paths=100_000, n_steps=100, seed=42, antithetic=True)
    mc = barrier_price_mc(S0, K, H, T, r, q, sigma, option=option, barrier=barrier, cfg=cfg)
    cf = _haug_barrier(S0, K, H, T, r, q, sigma, barrier, option)
    assert abs(mc - cf) < max(0.03, 0.01 * cf), f"MC {mc:.4f} vs closed form {cf:.4f}"


@pytest.mark.parametrize("barrier, option, K, H", _MC_CASES)
def test_mc_matches_closed_form_with_rebate(barrier, option, K, H):
    S0, T, r, q, sigma, rebate = 100.0, 1.0, 0.03, 0.01, 0.2, 3.0
    cfg = MCConfig(n_paths=100_000, n_steps=100, seed=7, antithetic=True)
    mc = barrier_price_mc(S0, K, H, T, r, q, sigma, option=option, barrier=barrier,
                          rebate=rebate, cfg=cfg)
    cf = _haug_barrier(S0, K, H, T, r, q, sigma, barrier, option, rebate=rebate)
    assert abs(mc - cf) < max(0.03, 0.01 * cf), f"MC {mc:.4f} vs closed form {cf:.4f}"
