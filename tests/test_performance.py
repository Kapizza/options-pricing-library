import math
import time
import numpy as np

from src.rough import rbergomi_paths, rough_heston_paths
from src.calibration import calibrate_rbergomi, calibrate_rough_heston


def _prices_from_ST(ST, r, T, strikes, cp="call"):
    DF = math.exp(-r * T)
    ST = np.asarray(ST, dtype=float).reshape(-1)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    if cp == "call":
        payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    else:
        payoff = np.maximum(K[None, :] - ST[:, None], 0.0)
    return DF * payoff.mean(axis=0)


def test_rbergomi_calibration_quick_performance():
    # Synthetic smile (one maturity)
    S0, r, q = 100.0, 0.01, 0.00
    T = 0.5
    strikes = np.array([80, 90, 100, 110, 120], float)
    cp = "call"
    H_true, eta_true, rho_true, xi0_true = 0.12, 1.40, -0.60, 0.04

    # Generate mids from a single MC set (CRN)
    t, S_paths, V_paths = rbergomi_paths(
        S0=S0, T=T, N=48, n_paths=2000,
        H=H_true, eta=eta_true, rho=rho_true, xi0=xi0_true,
        r=r, q=q, seed=202401, fgn_method="davies-harte"
    )
    ST = S_paths[:, -1]
    mids = _prices_from_ST(ST, r, T, strikes, cp=cp)
    smiles = [(S0, r, q, T, strikes, mids, cp)]

    # Quick calibration with thread backend and terminal-only
    t0 = time.time()
    best, _res = calibrate_rbergomi(
        smiles,
        metric="iv",
        vega_weight=True,
        x0=(0.11, 1.35, -0.55, 0.038),
        bounds=((0.05, 0.30), (0.4, 3.0), (-0.95, -0.05), (0.02, 0.08)),
        mc=dict(N=48, paths=2000, fgn_method="davies-harte", batch_size=4096, n_workers=4),
        multistart=1,
        options={"maxiter": 3},
        seed=202401,
        verbose=False,
        print_every=10,
        parallel_backend="thread",
        terminal_only=True,
    )
    dt = time.time() - t0
    # Very loose wall-clock bound to remain robust across environments
    assert dt < 12.0
    # Keep sanity on recovered parameters (very loose due to small maxiter)
    assert 0.04 < best["H"] < 0.45
    assert -0.99 < best["rho"] < -0.01


def test_rough_heston_calibration_quick_performance():
    # Synthetic smile (one maturity)
    S0, r, q = 100.0, 0.01, 0.00
    T = 0.6
    strikes = np.array([85, 95, 100, 105, 115], float)
    cp = "call"
    v0_true, kappa_true, theta_true = 0.04, 1.6, 0.04
    eta_true, rho_true, H_true = 1.8, -0.70, 0.10

    t, S_paths, V_paths = rough_heston_paths(
        S0=S0, v0=v0_true, T=T, N=48, n_paths=2000,
        H=H_true, kappa=kappa_true, theta=theta_true, eta=eta_true, rho=rho_true,
        r=r, q=q, seed=202402, batch_size=512
    )
    ST = S_paths[:, -1]
    mids = _prices_from_ST(ST, r, T, strikes, cp=cp)
    smiles = [(S0, r, q, T, strikes, mids, cp)]

    t0 = time.time()
    best, _res = calibrate_rough_heston(
        smiles,
        metric="price",
        vega_weight=False,
        x0=(0.035, 1.5, 0.035, 1.7, -0.6, 0.12),
        bounds=((0.005, 0.20), (0.1, 6.0), (0.005, 0.20), (0.2, 3.0), (-0.95, -0.05), (0.05, 0.45)),
        mc=dict(N=48, paths=2000, batch_size=512, n_workers=4),
        multistart=1,
        options={"maxiter": 3},
        verbose=False,
        print_every=10,
        parallel_backend="thread",
        terminal_only=True,
        seed=202402,
    )
    dt = time.time() - t0
    assert dt < 15.0
    # sanity
    assert 0.02 < best["H"] < 0.45
    assert -0.99 < best["rho"] < -0.01

