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


def _recover_WH_from_v(v, xi0, eta, t, H):
    # W_H(t) = [ ln(v/xi0) + 0.5*eta^2*t^{2H} ] / eta
    t2H = t**(2.0 * H)
    ln_term = np.log(np.maximum(v, 1e-300) / float(xi0))
    return (ln_term + 0.5 * (eta**2) * t2H[None, :]) / float(eta)


@pytest.fixture(scope="module")
def base():
    pars = dict(
        S0=100.0,
        T=0.75,
        N=192,
        n_paths=6000,   # keep moderate for speed; raise if needed
        H=0.10,
        eta=1.5,
        # The legacy fBM drivers below cannot carry spot-vol correlation and
        # now require rho=0. These tests only inspect v, which those drivers
        # generate independently of rho (identical v paths for rho=-0.7 and 0).
        rho=0.0,
        r=0.02,
        q=0.00,
        xi0=0.04,       # flat forward variance
        seed=123,
    )
    return pars


def _run(pars, method):
    t, S, v = rbergomi_paths(
        S0=pars["S0"], T=pars["T"], N=pars["N"], n_paths=pars["n_paths"],
        H=pars["H"], eta=pars["eta"], rho=pars["rho"], xi0=pars["xi0"],
        r=pars["r"], q=pars["q"], seed=pars["seed"], fgn_method=method
    )
    return t, S, v


def test_mean_variance_flat_under_flat_xi0_davies_harte(base):
    # Mean of v_t should be roughly flat around xi0 for flat xi0
    t, S, v = _run(base, "davies-harte")
    xi0 = base["xi0"]

    idxs = [int(0.1*base["N"]), int(0.5*base["N"]), int(0.9*base["N"])]
    mean_v = v.mean(axis=0)

    # tolerate a small MC bias; if this fails with very low n_paths, increase n_paths
    for k in idxs:
        assert abs(float(mean_v[k]) - xi0) < 4e-3, f"mean v deviates at t={t[k]:.3f}"


def test_wh_variance_matches_theory_davies_harte(base):
    # Var[W_H(t)] across paths should match t^{2H} within a reasonable band
    t, S, v = _run(base, "davies-harte")
    H = base["H"]
    xi0 = base["xi0"]
    eta = base["eta"]

    W = _recover_WH_from_v(v, xi0, eta, t, H)

    idxs = [int(0.1*base["N"]), int(0.5*base["N"]), int(0.9*base["N"])]
    for k in idxs:
        emp = float(np.var(W[:, k], ddof=1))
        theo = float(t[k]**(2.0 * H))
        # allow 15 percent relative error
        assert abs(emp - theo) <= 0.15 * max(1e-12, theo), f"k={k}, emp={emp}, theo={theo}"


def test_davies_harte_and_hosking_agree_on_means(base):
    # Compare FFT vs Hosking on the mean of v at end time
    t1, S1, v1 = _run(base, "davies-harte")
    t2, S2, v2 = _run(base, "hosking")

    m1 = float(v1.mean(axis=0)[-1])
    m2 = float(v2.mean(axis=0)[-1])
    # should be very close and both near xi0
    assert abs(m1 - base["xi0"]) < 5e-3
    assert abs(m2 - base["xi0"]) < 5e-3
    assert abs(m1 - m2) < 3e-3


def test_variance_is_stochastic_not_deterministic(base):
    # Cross-sectional variance of v at mid time should be non-trivial
    t, S, v = _run(base, "davies-harte")
    k = int(0.5 * base["N"])
    cross_var = float(np.var(v[:, k], ddof=1))
    assert cross_var > 1e-6, f"variance across paths too small: {cross_var}"


@pytest.mark.slow
def test_speed_branch_selection_visible(base):
    # This is a light speed check, not a strict benchmark. It confirms both methods run.
    # If you prefer to skip timing, keep this as a functional smoke test.
    t1, S1, v1 = _run(base, "davies-harte")
    t2, S2, v2 = _run(base, "hosking")
    assert S1.shape == S2.shape == (base["n_paths"], base["N"] + 1)
    assert v1.shape == v2.shape

# --------------------------
# Regression: Davies-Harte fBM has the exact fGn covariance
# --------------------------
from src.rough import fbm_davies_harte


@pytest.mark.parametrize("H", [0.1, 0.3, 0.7])
def test_davies_harte_covariance_is_exact(H):
    N, n_paths = 64, 100_000
    B = fbm_davies_harte(N, H, n_paths, np.random.default_rng(2024))
    X = np.diff(B, axis=1)                        # unit-step fGn
    gam = lambda k: 0.5 * (abs(k + 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k - 1) ** (2 * H))
    for lag in (0, 1, 2, 10):
        emp = float(np.mean(X[:, 5] * X[:, 5 + lag]))
        assert abs(emp - gam(lag)) < 0.02, f"lag {lag}: {emp} vs {gam(lag)}"
    for n in (1, 8, 64):
        emp = float(np.var(B[:, n]))
        assert abs(emp / n ** (2 * H) - 1.0) < 0.02, f"Var B({n}) = {emp} vs {n ** (2 * H)}"


# --------------------------
# Regression: rBergomi spot-vol correlation (hybrid Riemann-Liouville driver)
# --------------------------
from scipy.integrate import quad
from src.black_scholes import implied_vol_from_price


def _smile(S, K_list, T):
    ST = S[:, -1]
    return [implied_vol_from_price(100.0, K, T, 0.0, float(np.mean(np.maximum(ST - K, 0.0))), "call")
            for K in K_list]


def test_rbergomi_rho_controls_skew_and_correlation():
    T = 0.5
    skews, corrs = {}, {}
    for rho in (-0.9, 0.0, 0.9):
        t, S, v = rbergomi_paths(100.0, T, 64, 20000, 0.1, 1.9, rho, 0.04, seed=11)
        iv90, iv110 = _smile(S, (90.0, 110.0), T)
        skews[rho] = iv90 - iv110
        dlogS = np.diff(np.log(S), axis=1).ravel()
        dlogv = np.diff(np.log(v), axis=1).ravel()
        corrs[rho] = np.corrcoef(dlogS, dlogv)[0, 1]
    assert skews[-0.9] > 0.05 and skews[0.9] < -0.05 and abs(skews[0.0]) < 0.02
    # same-step corr is about rho * 0.42 here (old fBM driver: 0.004 for any rho)
    assert corrs[-0.9] < -0.25 and corrs[0.9] > 0.25 and abs(corrs[0.0]) < 0.02


def test_rbergomi_hybrid_forward_variance_and_volterra_variance():
    H, eta, xi0, T, N = 0.1, 1.0, 0.04, 0.5, 64
    t, S, v = rbergomi_paths(100.0, T, N, 40000, H, eta, -0.7, xi0, seed=5)
    for k in (8, 32, 64):
        assert abs(v[:, k].mean() / xi0 - 1.0) < 0.03
        W = (np.log(v[:, k] / xi0) + 0.5 * eta ** 2 * t[k] ** (2 * H)) / eta
        assert abs(np.var(W) / t[k] ** (2 * H) - 1.0) < 0.03


@pytest.mark.parametrize("method", ["davies-harte", "hosking"])
def test_legacy_fbm_drivers_reject_correlation(method):
    with pytest.raises(ValueError):
        rbergomi_paths(100.0, 0.5, 16, 10, 0.1, 1.5, -0.5, 0.04, seed=0, fgn_method=method)
    rbergomi_paths(100.0, 0.5, 16, 10, 0.1, 1.5, 0.0, 0.04, seed=0, fgn_method=method)


def _rbergomi_cholesky_calls(Ks, T, N, H, eta, rho, xi0, n_paths, seed):
    """Exact joint simulation of (W~_{t_i}, W_{t_i}) by Cholesky; same log-Euler spot step."""
    a = H - 0.5
    t = np.linspace(0.0, T, N + 1)[1:]
    C = np.zeros((2 * N, 2 * N))
    for i, u in enumerate(t):
        for j, w in enumerate(t):
            m = min(u, w)
            if i == j:
                C[i, j] = u ** (2 * H)
            elif u < w:
                C[i, j] = 2 * H * quad(lambda s: (w - s) ** a, 0, u, weight="alg", wvar=(0, a))[0]
            else:
                C[i, j] = 2 * H * quad(lambda s: (u - s) ** a, 0, w, weight="alg", wvar=(0, a))[0]
            C[i, N + j] = C[N + j, i] = math.sqrt(2 * H) / (a + 1) * (u ** (a + 1) - (u - m) ** (a + 1))
            C[N + i, N + j] = m
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_paths, 2 * N)) @ np.linalg.cholesky(C).T
    Y = np.hstack([np.zeros((n_paths, 1)), X[:, :N]])
    dW = np.diff(np.hstack([np.zeros((n_paths, 1)), X[:, N:]]), axis=1)
    tg = np.concatenate([[0.0], t])
    v = xi0 * np.exp(eta * Y - 0.5 * eta ** 2 * tg ** (2 * H))
    dt = T / N
    dB = rho * dW + math.sqrt(1 - rho * rho) * rng.standard_normal((n_paths, N)) * math.sqrt(dt)
    ST = 100.0 * np.exp(np.sum(np.sqrt(v[:, :-1]) * dB - 0.5 * v[:, :-1] * dt, axis=1))
    pays = [np.maximum(ST - K, 0.0) for K in Ks]
    return [p.mean() for p in pays], [p.std() / math.sqrt(n_paths) for p in pays]


def test_rbergomi_matches_exact_cholesky_simulation():
    # Independent reference: exact Gaussian simulation of the Volterra process
    # jointly with its driving Brownian motion (small grid).
    T, N, H, eta, rho, xi0, n = 0.5, 32, 0.1, 1.9, -0.9, 0.04, 200_000
    Ks = (80.0, 100.0, 120.0)
    ref, ref_se = _rbergomi_cholesky_calls(Ks, T, N, H, eta, rho, xi0, n, seed=1)
    t, S, v = rbergomi_paths(100.0, T, N, n, H, eta, rho, xi0, seed=2)
    for K, m, se in zip(Ks, ref, ref_se):
        pay = np.maximum(S[:, -1] - K, 0.0)
        mc, mc_se = pay.mean(), pay.std() / math.sqrt(n)
        assert abs(mc - m) < 4.0 * math.hypot(se, mc_se), f"K={K}: {mc:.4f} vs exact {m:.4f}"


def test_xi0_short_array_is_padded_with_last_value():
    from src.rough import _xi0_as_callable
    t = np.linspace(0.0, 1.0, 5)
    f = _xi0_as_callable([0.04, 0.09])
    assert np.allclose(f(t), [0.04, 0.09, 0.09, 0.09, 0.09])
    g = _xi0_as_callable([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    assert np.allclose(g(t), [0.01, 0.02, 0.03, 0.04, 0.05])
