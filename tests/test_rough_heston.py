# tests/test_rough_heston.py
import math
import numpy as np
import pytest

# Target under test
from src.rough import rough_heston_paths, rough_heston_euro_mc

# Optional classic Heston import for a soft consistency check
try:
    from src.heston import heston_price
    HAVE_HESTON = True
except Exception:
    HAVE_HESTON = False


# --------------------------
# Fixtures
# --------------------------

@pytest.fixture(scope="module")
def base_params():
    return dict(
        S0=100.0,
        v0=0.04,
        T=0.75,
        N=160,
        n_paths=3000,
        H=0.10,
        kappa=1.5,
        theta=0.04,
        eta=1.8,
        rho=-0.7,
        r=0.02,
        q=0.00,
        seed=123,
    )


# --------------------------
# Core pricing smoke tests
# --------------------------

def test_pricer_call_put_runs(base_params):
    p_call, se_call = rough_heston_euro_mc(**base_params, K=100.0, option="call")
    p_put, se_put   = rough_heston_euro_mc(**base_params, K=100.0, option="put")
    assert np.isfinite(p_call) and p_call > 0
    assert np.isfinite(p_put) and p_put >= 0
    assert se_call > 0 and se_put > 0


def test_put_call_parity(base_params):
    pars = base_params.copy()
    pars.update(dict(n_paths=6000, N=192, seed=7))
    K = 100.0
    c, se_c = rough_heston_euro_mc(**pars, K=K, option="call")
    p, se_p = rough_heston_euro_mc(**pars, K=K, option="put")
    rhs = pars["S0"] * math.exp(-pars["q"] * pars["T"]) - K * math.exp(-pars["r"] * pars["T"])
    lhs = c - p
    tol = 3.0 * math.sqrt(se_c**2 + se_p**2) + 1e-3
    assert abs(lhs - rhs) <= tol


def test_monotonic_in_strike_call(base_params):
    pars = base_params.copy()
    pars.update(dict(n_paths=2500, N=128, seed=11))
    strikes = [80, 90, 100, 110, 120]
    prices = [rough_heston_euro_mc(**pars, K=float(K), option="call")[0] for K in strikes]
    diffs = np.diff(prices)
    # Allow a tiny numerical wiggle
    assert np.all(diffs <= 2e-3)


# --------------------------
# Paths and positivity
# --------------------------

def test_paths_shapes_and_positivity(base_params):
    pars = base_params.copy()
    pars.update(dict(n_paths=256, N=96, seed=99))
    t, S, V = rough_heston_paths(**pars)
    assert t.shape == (pars["N"] + 1,)
    assert S.shape == (pars["n_paths"], pars["N"] + 1)
    assert V.shape == (pars["n_paths"], pars["N"] + 1)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(V))
    assert np.all(S > 0.0)
    assert np.all(V >= 0.0)


# --------------------------
# continuity checks in H
# --------------------------

def test_price_continuity_in_H():
    # Anchor parameters
    S0, K, T = 100.0, 105.0, 0.5
    r, q = 0.01, 0.0
    kappa, theta, eta, rho = 2.0, 0.09, 0.6, -0.4
    v0 = theta

    common = dict(
        S0=S0, v0=v0, K=K, T=T, N=256, n_paths=10000,
        kappa=kappa, theta=theta, eta=eta, rho=rho,
        r=r, q=q, option="call"
    )

    # Two nearby H values; same seed for variance reduction
    H1, H2 = 0.45, 0.49
    p1, se1 = rough_heston_euro_mc(H=H1, seed=321, **common)
    p2, se2 = rough_heston_euro_mc(H=H2, seed=321, **common)

    # Price difference should be within combined MC noise plus a small discretization cushion
    diff = abs(p1 - p2)
    mc_tol = 4.0 * math.sqrt(se1**2 + se2**2)  # generous, accounts for correlation
    disc_tol = 3e-2  # small cushion for time-discretization bias at N=256
    assert diff <= mc_tol + disc_tol

# --------------------------
# MC error scaling
# --------------------------

@pytest.mark.slow
def test_mc_error_decreases_with_paths(base_params):
    pars = base_params.copy()
    pars.update(dict(N=128, seed=5))

    pars_small = pars.copy()
    pars_small["n_paths"] = 1500

    pars_large = pars.copy()
    pars_large["n_paths"] = 6000

    p1, se1 = rough_heston_euro_mc(**pars_small, K=100.0, option="call")
    p2, se2 = rough_heston_euro_mc(**pars_large, K=100.0, option="call")

    assert se2 < se1
    ratio = se1 / se2
    assert ratio > 1.6  # ideal ~ sqrt(6000/1500) = 2.0


