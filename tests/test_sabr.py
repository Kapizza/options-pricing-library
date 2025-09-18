# tests/test_sabr.py
import numpy as np

from src.sabr import (
    sabr_iv, sabr_price, sabr_calibrate_iv, sabr_calibrate_price
)
from src.black_scholes import black_scholes_price, implied_vol_from_price


def test_sabr_put_call_parity():
    S, r, q = 100.0, 0.02, 0.01
    K, T = 100.0, 0.75
    pars = dict(alpha=0.25, beta=0.7, rho=-0.3, nu=0.6)

    c = sabr_price(S, K, T, r, q, **pars, option="call")
    p = sabr_price(S, K, T, r, q, **pars, option="put")

    DF = np.exp(-r*T)
    F  = S * np.exp((r-q)*T)
    lhs = c - p
    rhs = DF*(F - K)
    assert abs(lhs - rhs) < 1e-10


def test_sabr_reduces_to_bs_when_beta1_nu0_all_strikes():
    # With beta=1 and nu=0, Hagan IV collapses to alpha **for all K**.
    S, r, q = 100.0, 0.00, 0.00
    T = 1.25
    alpha, beta, rho, nu = 0.2, 1.0, -0.5, 0.0  # rho irrelevant when nu=0

    Ks = np.array([60, 80, 90, 100, 110, 140, 180], dtype=float)
    for K in Ks:
        # SABR price
        c_sabr = sabr_price(S, K, T, r, q, alpha, beta, rho, nu, option="call")
        p_sabr = sabr_price(S, K, T, r, q, alpha, beta, rho, nu, option="put")
        # BS with sigma=alpha
        c_bs = black_scholes_price(S, K, T, r, alpha, option_type="call")
        p_bs = black_scholes_price(S, K, T, r, alpha, option_type="put")
        assert abs(c_sabr - c_bs) < 1e-10
        assert abs(p_sabr - p_bs) < 1e-10


def test_sabr_iv_atm_continuity():
    # IV should be continuous at K=F
    F, T = 100.0, 0.9
    a, b, r, n = 0.3, 0.6, -0.4, 0.8
    K_atm = F
    K_lo  = F * (1 - 1e-8)
    K_hi  = F * (1 + 1e-8)

    iv_atm = sabr_iv(F, K_atm, T, a, b, r, n)
    iv_lo  = sabr_iv(F, K_lo,  T, a, b, r, n)
    iv_hi  = sabr_iv(F, K_hi,  T, a, b, r, n)

    # very tight because the closed-form has the K≈F limit built-in
    assert abs(iv_lo - iv_atm) < 1e-8
    assert abs(iv_hi - iv_atm) < 1e-8


def test_call_price_monotone_in_strike():
    # For fixed params, call price should (weakly) decrease as strike increases
    S, r, q = 100.0, 0.01, 0.0
    T = 0.5
    pars = dict(alpha=0.25, beta=0.7, rho=-0.2, nu=0.5)

    Ks = np.array([80, 90, 100, 110, 120], dtype=float)
    prices = [sabr_price(S, K, T, r, q, **pars, option="call") for K in Ks]
    assert all(prices[i] >= prices[i+1] - 1e-12 for i in range(len(prices)-1))


def test_iv_price_roundtrip_consistency():
    # Use SABR to price; invert with BS implied vol; re-price with BS and compare
    S, r, q = 100.0, 0.01, 0.0
    T = 1.0
    pars = dict(alpha=0.22, beta=0.8, rho=-0.3, nu=0.7)
    Ks = [80.0, 100.0, 130.0]

    for K in Ks:
        c_sabr = sabr_price(S, K, T, r, q, **pars, option="call")
        iv = implied_vol_from_price(S, K, T, r, c_sabr, option_type="call")
        c_bs = black_scholes_price(S, K, T, r, iv, option_type="call")
        assert abs(c_bs - c_sabr) < 1e-7


def test_calibration_price_fixed_beta_recovers_params():
    # Synthetic prices, calibrate in price space with beta fixed
    S, r, q = 100.0, 0.02, 0.0
    T  = 0.5
    alpha_t, beta_t, rho_t, nu_t = 0.20, 0.7, -0.4, 0.8

    Ks    = np.array([80, 90, 100, 110, 120], float)
    sides = np.array(["call", "call", "call", "put", "put"])

    # Generate clean synthetic prices
    prices = np.array([sabr_price(S, K, T, r, q, alpha_t, beta_t, rho_t, nu_t, option=sd)
                       for K, sd in zip(Ks, sides)])

    # Add mild noise
    rng = np.random.default_rng(1)
    prices_noisy = prices + rng.normal(0.0, 0.01, size=len(prices))

    # ---- vega weights (key change) ----
    # Get a proxy IV at each strike from the true SABR to compute BS vega
    F  = S * np.exp((r - q) * T)
    iv = np.array([sabr_iv(F, K, T, alpha_t, beta_t, rho_t, nu_t) for K in Ks])

    # BS vega in spot-measure: S * phi(d1) * sqrt(T)
    def bs_vega(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
        return S * (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*d1*d1) * np.sqrt(T)

    w = np.array([bs_vega(S, K, T, r, sig) for K, sig in zip(Ks, iv)])
    # normalize to avoid scale issues
    w = w / max(w.max(), 1e-12)

    # Provide a decent initial guess to avoid bad local minima
    x0 = np.array([0.22, -0.35, 0.7])  # alpha, rho, nu (beta fixed)

    params, res = sabr_calibrate_price(S, r, q, Ks, np.full_like(Ks, T),
                                       prices_noisy, sides, w=w, beta=beta_t,
                                       x0=x0, bounds=None, maxiter=400, disp=False)

    assert abs(params["alpha"] - alpha_t) < 0.03
    assert abs(params["rho"]   - rho_t)   < 0.15
    assert abs(params["nu"]    - nu_t)    < 0.15
    assert abs(params["beta"]  - beta_t)  < 1e-12


def test_calibration_price_fixed_beta_recovers_params():

    S, r, q = 100.0, 0.02, 0.0
    T  = 0.5
    alpha_t, beta_t, rho_t, nu_t = 0.20, 0.7, -0.4, 0.8

    Ks    = np.array([80, 90, 100, 110, 120], float)
    sides = np.array(["call", "call", "call", "put", "put"])

    prices = np.array([sabr_price(S, K, T, r, q, alpha_t, beta_t, rho_t, nu_t, option=sd)
                       for K, sd in zip(Ks, sides)])

    rng = np.random.default_rng(1)
    prices_noisy = prices + rng.normal(0.0, 0.01, size=len(prices))

    # --- vega weights (proxy from true SABR IV) ---
    F  = S * np.exp((r - q) * T)
    iv = np.array([sabr_iv(F, K, T, alpha_t, beta_t, rho_t, nu_t) for K in Ks])

    def bs_vega(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0: return 0.0
        d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
        return S * (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*d1*d1) * np.sqrt(T)

    w = np.array([bs_vega(S, K, T, r, sig) for K, sig in zip(Ks, iv)])
    w = w / max(w.max(), 1e-12)

    # helpful initial guess near truth
    x0 = np.array([0.22, -0.35, 0.75])  # alpha, rho, nu (beta fixed)

    params, res = sabr_calibrate_price(
        S, r, q, Ks, np.full_like(Ks, T),
        prices_noisy, sides, w=w, beta=beta_t,
        x0=x0, bounds=[(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)],
        maxiter=400, disp=False
    )

    assert abs(params["alpha"] - alpha_t) < 0.03
    assert abs(params["rho"]   - rho_t)   < 0.12
    assert abs(params["nu"]    - nu_t)    < 0.12
    assert abs(params["beta"]  - beta_t)  < 1e-12
