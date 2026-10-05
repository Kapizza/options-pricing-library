# tests/test_calibration.py
import math
import numpy as np
import pytest


from src.calibration import calibrate_rbergomi, calibrate_rough_heston

from src.rough import rbergomi_paths, rbergomi_terminal_parallel_pool
from concurrent.futures import ThreadPoolExecutor
from src.rough import rough_heston_paths


def _prices_from_ST(ST, r, T, strikes, cp="call"):
    DF = math.exp(-r * T)
    out = []
    for K in strikes:
        if cp == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        out.append(float(np.mean(DF * payoff)))
    return np.array(out, dtype=float)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_rbergomi_calibration_recovers_params_iv():
    # --- synthetic market data ---
    S0, r, q = 100.0, 0.01, 0.00
    T = 0.5
    strikes = np.linspace(70, 130, 13, dtype=float)  # more strikes → better smile shape
    cp = "call"
    H_true, eta_true, rho_true, xi0_true = 0.12, 1.40, -0.60, 0.04

    # generate one MC set and reuse for all strikes (CRN). The calibration
    # simulates with base_seed = seed + int(1000*T) split into batches, so the
    # market must be drawn the same way for the random numbers to be common.
    seed_mkt = 2024
    with ThreadPoolExecutor(max_workers=4) as ex:
        ST = rbergomi_terminal_parallel_pool(
            ex, S0=S0, T=T, N=128, n_paths=6000,
            H=H_true, eta=eta_true, rho=rho_true, xi0=xi0_true,
            r=r, q=q, base_seed=seed_mkt + int(1000 * T), fgn_method="hybrid", batch_size=750
        )
    mids = _prices_from_ST(ST, r, T, strikes, cp=cp)
    smiles = [(S0, r, q, T, strikes, mids, cp)]

    # --- calibrate in IV space with vega weights; use same seed for CRN ---
    best, _res = calibrate_rbergomi(
        smiles,
        metric="iv",
        vega_weight=True,
        x0=(0.11, 1.35, -0.55, 0.038),                 # close-ish start
        bounds=((0.05, 0.30), (0.4, 3.0), (-0.95, -0.05), (0.02, 0.08)),  # keep H off edges
        mc=dict(N=128, paths=6000, fgn_method="hybrid", batch_size=750, n_workers=4),
        multistart=2,
        options={"maxiter": 80},
        seed=seed_mkt,                                   # CRN with market mids
        verbose=False,
        print_every=20,
        parallel_backend="thread",
    )

    # tolerances are still loose to allow MC noise
    assert abs(best["H"]   - H_true)   < 0.05
    assert abs(best["eta"] - eta_true) < 0.30
    assert abs(best["rho"] - rho_true) < 0.12
    assert abs(best["xi0"] - xi0_true) < 0.01


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_rough_heston_calibration_recovers_params_price():
    # --- synthetic market data ---
    S0, r, q = 100.0, 0.01, 0.00
    T = 0.75
    strikes = np.array([85, 95, 100, 105, 115], dtype=float)
    cp = "call"
    # true params
    v0_true, kappa_true, theta_true = 0.04, 1.6, 0.04
    eta_true, rho_true, H_true = 1.8, -0.70, 0.10

    t, S_paths, V_paths = rough_heston_paths(
        S0=S0, v0=v0_true, T=T, N=96, n_paths=3500,
        H=H_true, kappa=kappa_true, theta=theta_true, eta=eta_true, rho=rho_true,
        r=r, q=q, seed=2025, batch_size=512
    )
    ST = S_paths[:, -1]
    mids = _prices_from_ST(ST, r, T, strikes, cp=cp)
    smiles = [(S0, r, q, T, strikes, mids, cp)]

    # --- calibrate (price space) ---
    best, _res = calibrate_rough_heston(
        smiles,
        metric="price",
        vega_weight=False,
        x0=(0.035, 1.5, 0.035, 1.7, -0.6, 0.12),
        bounds=((0.005, 0.20), (0.1, 6.0), (0.005, 0.20), (0.2, 3.0), (-0.95, -0.05), (0.05, 0.45)),
        mc=dict(N=96, paths=3500, batch_size=512),
        multistart=1,
        options={"maxiter": 45},
        verbose=False,
        print_every=10,
        parallel_backend="thread",
    )

    # --- checks: loose band due to MC noise and many params ---
    assert abs(best["v0"]    - v0_true)    < 0.015
    assert abs(best["kappa"] - kappa_true) < 0.5
    assert abs(best["theta"] - theta_true) < 0.015
    assert abs(best["eta"]   - eta_true)   < 0.50
    assert abs(best["rho"]   - rho_true)   < 0.20
    assert abs(best["H"]     - H_true)     < 0.06


def test_smoke_iv_mode_and_progress_history():
    # small smoke test to ensure IV mode runs and history gets attached
    S0, r, q = 100.0, 0.01, 0.0
    T = 0.4
    K = np.array([90, 100, 110], float)
    cp = "call"
    H_true, eta_true, rho_true, xi0_true = 0.11, 1.2, -0.5, 0.04

    t, S_paths, V_paths = rbergomi_paths(
        S0=S0, T=T, N=64, n_paths=2500,
        H=H_true, eta=eta_true, rho=rho_true, xi0=xi0_true,
        r=r, q=q, seed=999, fgn_method="hybrid"
    )
    ST = S_paths[:, -1]
    mids = _prices_from_ST(ST, r, T, K, cp=cp)
    smiles = [(S0, r, q, T, K, mids, cp)]

    best, res = calibrate_rbergomi(
        smiles,
        metric="iv",
        vega_weight=True,
        x0=(0.10, 1.3, -0.45, 0.035),
        mc=dict(N=64, paths=2500, fgn_method="hybrid"),
        multistart=1,
        options={"maxiter": 12},
        verbose=True,        # exercise the monitor
        print_every=2,
        parallel_backend="thread",
    )

    assert "history" in best and isinstance(best["history"], list)
    # iteration history should have at least one item if maxiter > 0
    assert len(best["history"]) >= 1


# ---------------------------------------------------------------------------
# Regression: market/model IV inversion with a dividend yield
# ---------------------------------------------------------------------------
from src.calibration import _iv_or_nan
from src.black_scholes import black_scholes_price


@pytest.mark.parametrize("cp", ["call", "put"])
def test_iv_or_nan_roundtrip_with_dividends(cp):
    S, T, r, q, sigma = 100.0, 0.5, 0.04, 0.03, 0.25
    for K in (80.0, 100.0, 120.0):
        px = black_scholes_price(S, K, T, r, sigma, cp, q=q)
        assert abs(_iv_or_nan(S, K, T, r, q, px, cp) - sigma) < 1e-5


def test_iv_or_nan_accepts_deep_itm_put_below_intrinsic():
    S, K, T, r, sigma = 60.0, 100.0, 1.0, 0.05, 0.2
    px = black_scholes_price(S, K, T, r, sigma, "put")
    assert abs(_iv_or_nan(S, K, T, r, 0.0, px, "put") - sigma) < 1e-5


# ---------------------------------------------------------------------------
# Regression: the MC calibrators' relative finite-difference step is honoured.
# SciPy's L-BFGS-B passes `eps` (1e-8) as an absolute step, which overrode the
# intended 5% relative step, so gradients of the MC objective were taken over
# 1e-8 parameter moves.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["rbergomi", "rough_heston"])
def test_mc_calibrators_use_relative_fd_steps(model, monkeypatch):
    import src.calibration as cal
    S0, r, q, T = 100.0, 0.01, 0.0, 0.5
    K = np.array([90.0, 100.0, 110.0])
    mids = np.array([11.0, 5.5, 2.0])
    seen = []
    if model == "rbergomi":
        name, fn = "_rbergomi_objective", cal.calibrate_rbergomi
        x0 = np.array([0.12, 1.4, -0.6, 0.04])
        kw = dict(mc=dict(N=16, paths=400, fgn_method="hybrid"))
    else:
        name, fn = "_rough_heston_objective", cal.calibrate_rough_heston
        x0 = np.array([0.04, 1.5, 0.04, 1.0, -0.6, 0.12])
        kw = dict(mc=dict(N=16, paths=400, batch_size=400))
    orig = getattr(cal, name)

    def recording(params, *a, **k):
        seen.append(np.array(params, dtype=float))
        return orig(params, *a, **k)

    monkeypatch.setattr(cal, name, recording)
    fn([(S0, r, q, T, K, mids, "call")], metric="price", vega_weight=False, x0=tuple(x0),
       multistart=1, options={"maxiter": 1}, verbose=False, parallel_backend="thread",
       n_workers=1, **kw)
    probes = seen[1:1 + len(x0)]          # forward-difference probes around x0
    for p in probes:
        moved = np.flatnonzero(p != x0)
        assert moved.size == 1
        i = moved[0]
        assert 0.04 < abs(p[i] - x0[i]) / abs(x0[i]) < 0.06
