# tests/test_rough.py
import math
import numpy as np
import pytest

from src.rough import (
    rbergomi_euro_mc,
    rbergomi_paths,
    fbm_increments_hosking,
)
from src.black_scholes import black_scholes_price


# --------------------------
# Fixtures / common params
# --------------------------

@pytest.fixture(scope="module")
def base_params():
    return dict(
        S0=100.0,
        K=100.0,
        T=0.75,
        r=0.02,
        q=0.00,
        H=0.10,
        eta=1.50,
        rho=-0.7,
    )


# --------------------------
# Core pricing smoke tests
# --------------------------

def test_pricer_call_put_runs(base_params):
    p_call, se_call = rbergomi_euro_mc(
        **base_params, xi0=0.04, n_paths=4000, N=200, option="call", seed=42
    )
    p_put, se_put = rbergomi_euro_mc(
        **base_params, xi0=0.04, n_paths=4000, N=200, option="put", seed=42
    )
    assert np.isfinite(p_call) and p_call > 0
    assert np.isfinite(p_put) and p_put >= 0
    assert se_call > 0 and se_put > 0


def test_put_call_parity(base_params):
    # Parity: C - P = S0*e^{-qT} - K*e^{-rT}
    S0 = base_params["S0"]
    K = base_params["K"]
    T = base_params["T"]
    r = base_params["r"]
    q = base_params["q"]

    c, se_c = rbergomi_euro_mc(**base_params, xi0=0.04, n_paths=8000, N=256, option="call", seed=7)
    p, se_p = rbergomi_euro_mc(**base_params, xi0=0.04, n_paths=8000, N=256, option="put", seed=7)

    rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    lhs = c - p
    # Tolerance based on combined MC error (~ 3 sigma)
    tol = 3.0 * math.sqrt(se_c**2 + se_p**2)
    assert abs(lhs - rhs) <= max(1e-4, tol)


def test_monotonic_in_strike_call(base_params):
    # Call price should decrease as K increases, other things equal
    params = base_params.copy()
    prices = []
    for K in [80, 90, 100, 110, 120]:
        params["K"] = float(K)
        price, _ = rbergomi_euro_mc(**params, xi0=0.04, n_paths=3000, N=180, option="call", seed=11)
        prices.append(price)
    diffs = np.diff(prices)
    # Allow tiny numerical noise
    assert np.all(diffs <= 1e-3)


# --------------------------
# Reduction to Black–Scholes when eta=0
# --------------------------

def test_reduction_to_bs_eta_zero():
    # With eta=0 => v_t = xi0(t) constant; rough params irrelevant; equals BS
    S0, K, T, r, q = 100.0, 105.0, 0.5, 0.01, 0.0
    xi0 = 0.09  # variance -> sigma = 0.3
    sigma = math.sqrt(xi0)

    c_mc, se = rbergomi_euro_mc(
        S0=S0, K=K, T=T, r=r, q=q, H=0.3, eta=0.0, rho=0.0, xi0=xi0,
        n_paths=12000, N=100, option="call", seed=123
    )
    c_bs = black_scholes_price(S0, K, T, r, sigma, option_type="call", q=q)
    # Expect equality within MC error plus a small discretization cushion
    assert abs(c_mc - c_bs) <= max(2.5 * se, 1e-3)


# --------------------------
# xi0 handling (scalar, array, callable)
# --------------------------

def test_xi0_variants(base_params):
    t_steps = 128
    n_paths = 4000

    # Scalar xi0
    p_scalar, _ = rbergomi_euro_mc(
        **base_params, xi0=0.04, n_paths=n_paths, N=t_steps, option="call", seed=1
    )

    # Array xi0 (flat)
    xi_arr = np.full(t_steps + 1, 0.04)
    p_array, _ = rbergomi_euro_mc(
        **base_params, xi0=xi_arr, n_paths=n_paths, N=t_steps, option="call", seed=1
    )

    # Callable xi0 (mild upward term-structure)
    def xi_callable(t):
        # 4% at start, +2 vol-pts of variance linearly to T (i.e., 0.06 variance at T)
        return 0.04 + 0.02 * (t / t[-1] if t[-1] > 0 else 0.0)

    p_callable, _ = rbergomi_euro_mc(
        **base_params, xi0=xi_callable, n_paths=n_paths, N=t_steps, option="call", seed=1
    )

    # Flat scalar vs flat array should be essentially equal
    assert abs(p_scalar - p_array) <= 3e-3
    # Callable with higher late variance should not be lower than flat price
    assert p_callable >= p_scalar - 3e-3


# --------------------------
# Paths and variance sanity
# --------------------------

def test_paths_shapes_and_positivity(base_params):
    keep = ["S0", "T", "r", "q", "H", "eta", "rho"]
    pars = {k: base_params[k] for k in keep}
    t, S, v = rbergomi_paths(
        **pars, xi0=0.04, n_paths=512, N=64, seed=99
    )
    assert t.shape == (65,)
    assert S.shape == (512, 65)
    assert v.shape == (512, 65)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0.0)
    # Spot should be positive
    assert np.all(S > 0.0)


# --------------------------
# fBM generator sanity
# --------------------------

def test_fbm_increments_hosking_basic():
    H = 0.2
    N = 2000
    rng = np.random.default_rng(123)
    x = fbm_increments_hosking(N, H, rng=rng)
    assert x.shape == (N,)
    # zero mean approx
    assert abs(np.mean(x)) < 0.1
    # variance near 1 for unit-step fGn
    assert 0.7 < np.var(x) < 1.3


# --------------------------
# MC error scaling
# --------------------------

@pytest.mark.slow
def test_mc_error_decreases_with_paths(base_params):
    # Larger n_paths should reduce stderr roughly by sqrt-ratio
    p1, se1 = rbergomi_euro_mc(
        **base_params, xi0=0.04, n_paths=2000, N=160, option="call", seed=5
    )
    p2, se2 = rbergomi_euro_mc(
        **base_params, xi0=0.04, n_paths=8000, N=160, option="call", seed=5
    )
    # allow some randomness; check monotone decrease and close to sqrt scaling
    assert se2 < se1
    ratio = se1 / se2
    assert ratio > 1.6  # sqrt(4) = 2.0, allow slack


# --------------------------
# Parameter guardrails
# --------------------------

def test_bad_params_raise():
    with pytest.raises(ValueError):
        rbergomi_paths(S0=100, T=1.0, N=0, n_paths=100, H=0.1, eta=1.0, rho=0.0, xi0=0.04)

    with pytest.raises(ValueError):
        rbergomi_paths(S0=100, T=1.0, N=10, n_paths=0, H=0.1, eta=1.0, rho=0.0, xi0=0.04)

    with pytest.raises(ValueError):
        rbergomi_paths(S0=100, T=1.0, N=10, n_paths=10, H=-0.1, eta=1.0, rho=0.0, xi0=0.04)

    with pytest.raises(ValueError):
        rbergomi_paths(S0=100, T=1.0, N=10, n_paths=10, H=0.1, eta=1.0, rho=1.2, xi0=0.04)

    with pytest.raises(ValueError):
        fbm_increments_hosking(N=10, H=1.1)

def test_option_flag_validation(base_params):
    # invalid option string should raise
    with pytest.raises(ValueError):
        rbergomi_euro_mc(**base_params, xi0=0.04, n_paths=1000, N=64, option="CALLL", seed=0)


# --------------------------
# Very light regression anchor (non brittle)
# --------------------------

def test_regression_anchor_small_seeded(base_params):
    # Anchor a tiny seeded run to catch major regressions without being brittle.
    price, se = rbergomi_euro_mc(
        **base_params, xi0=0.04, n_paths=1500, N=96, option="call", seed=321
    )
    # Just check sane range and that CI is not absurdly wide.
    assert 1.0 < price < 25.0
    assert se < 1.0
