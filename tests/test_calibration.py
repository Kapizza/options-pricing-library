# tests/test_calibration.py
import math
import numpy as np
import pytest


from src.calibration import calibrate_rbergomi, calibrate_rough_heston

from src.rough import rbergomi_paths
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

    # generate one MC set and reuse for all strikes (CRN)
    seed_mkt = 2024
    t, S_paths, V_paths = rbergomi_paths(
        S0=S0, T=T, N=128, n_paths=6000,
        H=H_true, eta=eta_true, rho=rho_true, xi0=xi0_true,
        r=r, q=q, seed=seed_mkt, fgn_method="davies-harte"
    )
    ST = S_paths[:, -1]
    mids = _prices_from_ST(ST, r, T, strikes, cp=cp)
    smiles = [(S0, r, q, T, strikes, mids, cp)]

    # --- calibrate in IV space with vega weights; use same seed for CRN ---
    best, _res = calibrate_rbergomi(
        smiles,
        metric="iv",
        vega_weight=True,
        x0=(0.11, 1.35, -0.55, 0.038),                 # close-ish start
        bounds=((0.05, 0.30), (0.4, 3.0), (-0.95, -0.05), (0.02, 0.08)),  # keep H off edges
        mc=dict(N=128, paths=6000, fgn_method="davies-harte"),
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
        r=r, q=q, seed=999, fgn_method="davies-harte"
    )
    ST = S_paths[:, -1]
    mids = _prices_from_ST(ST, r, T, K, cp=cp)
    smiles = [(S0, r, q, T, K, mids, cp)]

    best, res = calibrate_rbergomi(
        smiles,
        metric="iv",
        vega_weight=True,
        x0=(0.10, 1.3, -0.45, 0.035),
        mc=dict(N=64, paths=2500, fgn_method="davies-harte"),
        multistart=1,
        options={"maxiter": 12},
        verbose=True,        # exercise the monitor
        print_every=2,
        parallel_backend="thread",
    )

    assert "history" in best and isinstance(best["history"], list)
    # iteration history should have at least one item if maxiter > 0
    assert len(best["history"]) >= 1
