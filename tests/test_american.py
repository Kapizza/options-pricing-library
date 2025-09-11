
import math
import numpy as np
from src.american import (
    american_binomial,
    american_lsmc,
    american_put_lsmc,
    american_call_lsmc,
    american_price,
)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.black_scholes import black_scholes_price

# --- Helpers ---

def european_put_price(S, K, T, r, sigma):
    return black_scholes_price(S, K, T, r, sigma, option_type="put")

def european_call_price(S, K, T, r, sigma):
    return black_scholes_price(S, K, T, r, sigma, option_type="call")


# --- Tests ---

def test_binomial_call_matches_bs_no_dividends():
    # American call with no dividends should equal European call
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_call = european_call_price(S, K, T, r, sigma)
    am_call = american_binomial(S, K, T, r, sigma, steps=800, option="call")
    assert abs(am_call - bs_call) < 1e-2  # tight with many steps


def test_binomial_put_vs_lsmc_close():
    # American put: LSMC should be close to binomial (allow approximation slack)
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.25
    am_bin = american_binomial(S, K, T, r, sigma, steps=800, option="put")
    am_lsmc = american_put_lsmc(S, K, T, r, sigma, n_paths=120_000, n_steps=75, seed=1)
    # LSMC is noisy; allow moderate tolerance (tighten with more paths)
    assert abs(am_lsmc - am_bin) < 0.5


def test_lsmc_put_ge_european_put():
    # American >= European for puts
    S, K, T, r, sigma = 100.0, 100.0, 0.75, 0.03, 0.30
    eu_put = european_put_price(S, K, T, r, sigma)
    am_put = american_lsmc(S, K, T, r, sigma, option="put", n_paths=80_000, n_steps=60, seed=2)
    assert am_put >= eu_put - 1e-3


def test_lsmc_seed_reproducibility():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    v1 = american_call_lsmc(S, K, T, r, sigma, n_paths=50_000, n_steps=50, seed=123)
    v2 = american_call_lsmc(S, K, T, r, sigma, n_paths=50_000, n_steps=50, seed=123)
    assert v1 == v2  # exact reproducibility for fixed seed and same params


def test_dispatch_methods():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    v_lsmc = american_price(S, K, T, r, sigma, method="lsmc", option="put", n_paths=20_000, n_steps=40, seed=0)
    v_bin  = american_price(S, K, T, r, sigma, method="binomial", option="put", steps=400)
    assert v_lsmc > 0 and v_bin > 0
    # They should be in the same ballpark
    assert abs(v_lsmc - v_bin) < 1.0


def test_invalid_option_raises():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    import pytest
    with pytest.raises(ValueError):
        american_lsmc(S, K, T, r, sigma, option="invalid", n_paths=10_000, n_steps=20, seed=0)
