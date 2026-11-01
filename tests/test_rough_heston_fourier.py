# tests/test_rough_heston_fourier.py
import numpy as np
import pytest

from src.rough_heston_fourier import rough_heston_charfunc, rough_heston_price, rough_heston_smile_prices
from src.heston import heston_smile_prices
from src.rough import rough_heston_paths

P = dict(kappa=1.5, theta=0.04, eta=0.5, v0=0.05, rho=-0.7)
K = np.array([70.0, 85.0, 100.0, 115.0, 130.0])


@pytest.mark.parametrize("T", [0.25, 1.0, 2.0])
def test_h_one_half_is_classical_heston(T):
    # At H = 1/2 the fractional Riccati equation is Heston's Riccati ODE; the
    # implicit Adams scheme is second order (max error 6.4e-7 at T=2, n=800).
    ref = heston_smile_prices(100.0, 0.02, 0.01, T, K, sigma=P["eta"], kappa=P["kappa"],
                              theta=P["theta"], v0=P["v0"], rho=P["rho"], N=4096)
    for opt in ("call", "put"):
        got = rough_heston_smile_prices(100.0, 0.02, 0.01, T, K, H=0.5, n_steps=800, option=opt, **P)
        want = ref if opt == "call" else ref - 100.0 * np.exp(-0.01 * T) + K * np.exp(-0.02 * T)
        assert np.max(np.abs(got - want)) < 2e-6


def test_charfunc_basics():
    assert rough_heston_charfunc(0.0, 1.0, 0.0, H=0.1, S0=100.0, **P) == 1.0
    u = np.linspace(0.1, 50.0, 30)
    phi = rough_heston_charfunc(u, 1.0, 0.0, H=0.1, S0=1.0, **P)
    assert np.all(np.abs(phi) <= 1.0 + 1e-12)
    # conjugate symmetry phi(-u) = conj(phi(u))
    assert np.allclose(rough_heston_charfunc(-u, 1.0, 0.0, H=0.1, S0=1.0, **P), np.conj(phi), atol=1e-12)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3])
def test_converges_in_time_steps(H):
    p200 = rough_heston_smile_prices(100.0, 0.0, 0.0, 1.0, K, H=H, n_steps=200, **P)
    p800 = rough_heston_smile_prices(100.0, 0.0, 0.0, 1.0, K, H=H, n_steps=800, **P)
    assert np.max(np.abs(p200 - p800)) < 2e-4


def test_parity_and_no_arbitrage_bounds():
    S0, r, q = 100.0, 0.03, 0.01
    for T in (0.05, 0.5, 2.0):
        KK = S0 * np.exp(np.linspace(-0.6, 0.5, 23))
        c = rough_heston_smile_prices(S0, r, q, T, KK, H=0.1, option="call", **P)
        p = rough_heston_smile_prices(S0, r, q, T, KK, H=0.1, option="put", **P)
        fwd = S0 * np.exp(-q * T) - KK * np.exp(-r * T)
        assert np.allclose(c - p, fwd, atol=1e-10)
        assert np.all(c >= np.maximum(fwd, 0.0) - 1e-8) and np.all(c <= S0 * np.exp(-q * T) + 1e-8)
        assert np.all(np.diff(c) <= 1e-10)                   # decreasing in K
    assert rough_heston_price(100.0, 100.0, 1.0, 0.0, H=0.1, **P) == pytest.approx(
        rough_heston_smile_prices(100.0, 0.0, 0.0, 1.0, [100.0], H=0.1, **P)[0])


def test_matches_monte_carlo_at_low_vol_of_vol():
    # Independent check against the Volterra Euler Monte Carlo. At eta = 0.1 the
    # variance rarely reaches the scheme's floor, so its bias is small (at
    # eta = 0.5 the MC scheme is biased upward: ATM 7.83 / 7.65 at N = 100 / 400
    # steps, vs 7.39 here, moving toward it as N grows).
    p = dict(P, eta=0.1)
    fourier = rough_heston_smile_prices(100.0, 0.0, 0.0, 1.0, K, H=0.1, n_steps=400, **p)
    _, S, _ = rough_heston_paths(100.0, p["v0"], 1.0, 200, 60000, 0.1, p["kappa"], p["theta"],
                                 p["eta"], p["rho"], seed=7, batch_size=20000)
    pay = np.maximum(S[:, -1][:, None] - K[None, :], 0.0)
    c, se = pay.mean(axis=0), pay.std(axis=0, ddof=1) / np.sqrt(pay.shape[0])
    assert np.all(np.abs(c - fourier) < 3.0 * se)


def test_fourier_calibration_recovers_parameters_from_far_start():
    from src.calibration import calibrate_rough_heston_fourier
    truth = dict(v0=0.05, kappa=1.5, theta=0.04, eta=0.5, rho=-0.7, H=0.1)
    num = dict(cos_N=256, cos_L=12.0, n_steps=100)
    smiles = []
    for T in (0.1, 0.5, 2.0):
        KK = 100.0 * np.exp(np.linspace(-0.25, 0.2, 7) * np.sqrt(T / 0.5))
        mids = rough_heston_smile_prices(100.0, 0.02, 0.01, T, KK, N=256, n_steps=100, **truth)
        smiles.append((100.0, 0.02, 0.01, T, KK, mids, "call"))
    # start outside every tolerance of the Monte Carlo recovery test
    best, _ = calibrate_rough_heston_fourier(
        smiles, metric="price", vega_weight=False, x0=(0.03, 0.8, 0.06, 1.0, -0.4, 0.3),
        numerics=num, multistart=1, verbose=False)
    assert best["obj"] < 1e-10
    for k, v in truth.items():
        assert abs(best[k] - v) < 1e-3 * max(1.0, abs(v)), k
