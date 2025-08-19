# tests/test_monte_carlo.py
import numpy as np
from options_pricing.monte_carlo import monte_carlo_option_pricing
from options_pricing.black_scholes import black_scholes_price

def test_monte_carlo_call_close_to_bs():
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma, option_type="call", n_simulations=200000, seed=42)
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type="call")
    assert np.isclose(mc_price, bs_price, rtol=0.02)  # within 2%

def test_monte_carlo_put_close_to_bs():
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma, option_type="put", n_simulations=200000, seed=42)
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type="put")
    assert np.isclose(mc_price, bs_price, rtol=0.02)  # within 2%
