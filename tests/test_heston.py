
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.heston import heston_price, heston_charfunc, heston_call_put
from src.black_scholes import black_scholes_price

def test_charfunc_normalization():
    # phi(0) must be 1
    v = heston_charfunc(0.0, T=1.0, r=0.03, kappa=2.0, theta=0.04, sigma=0.2, v0=0.04, rho=-0.5)
    assert abs(v - 1.0) < 1e-12

def test_put_call_parity():
    S0, K, T, r = 100.0, 100.0, 1.0, 0.03
    kappa, theta, sigma, v0, rho = 2.0, 0.04, 0.5, 0.04, -0.7
    c = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="call")
    p = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="put")
    lhs = c - p
    rhs = S0 - K * np.exp(-r*T)
    assert abs(lhs - rhs) < 1e-6

def test_heston_approaches_bs_in_low_volofvol_limit():
    # When variance is (nearly) constant: kappa large, sigma small, v0 ~= theta, rho ~= 0
    S0, K, T, r = 100.0, 100.0, 1.0, 0.01
    iv = 0.2
    theta = iv**2
    params = dict(kappa=8.0, theta=theta, sigma=1e-3, v0=theta, rho=0.0)
    c_heston = heston_price(S0, K, T, r, **params, option="call", alpha=1.5, N=4096, umax=200.0)
    c_bs     = black_scholes_price(S0, K, T, r, iv, option_type="call")
    assert abs(c_heston - c_bs) < 5e-3  # tight since transform is stable here

def test_strike_monotonicity_calls():
    # Call price decreases with strike (weakly) for fixed params
    S0, T, r = 100.0, 1.0, 0.01
    params = dict(kappa=2.0, theta=0.04, sigma=0.5, v0=0.04, rho=-0.5)
    Ks = [80, 90, 100, 110, 120]
    prices = [heston_price(S0, K, T, r, **params, option="call") for K in Ks]
    assert all(prices[i] >= prices[i+1] - 1e-8 for i in range(len(prices)-1))
