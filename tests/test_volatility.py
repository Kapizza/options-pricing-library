# tests/test_volatility.py

import numpy as np
import pytest
from src.volatility import implied_volatility, implied_vol_surface
from src.black_scholes import black_scholes_price


@pytest.mark.parametrize("sigma_true, option_type", [
    (0.1, "call"),
    (0.2, "call"),
    (0.3, "put"),
    (0.5, "put"),
])
def test_implied_volatility_parametrized(sigma_true, option_type):
    S, K, T, r = 100, 100, 1.0, 0.05
    price = black_scholes_price(S, K, T, r, sigma_true, option_type)

    sigma_est = implied_volatility(price, S, K, T, r, option_type)
    assert np.isclose(sigma_est, sigma_true, atol=1e-4)


def test_implied_volatility_invalid_price():
    # Negative or zero option price should raise
    with pytest.raises(ValueError):
        implied_volatility(0.0, 100, 100, 1.0, 0.05)


def test_implied_vol_surface():
    S, r = 100, 0.01
    strikes = np.array([90, 100, 110])
    maturities = np.array([0.5, 1.0])
    sigma_true = 0.25

    # Generate a simple price grid from BS
    prices = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            prices[i, j] = black_scholes_price(S, K, T, r, sigma_true, "call")

    surface = implied_vol_surface(prices, S, strikes, maturities, r, "call")

    assert surface.shape == (len(maturities), len(strikes))
    assert np.allclose(surface, sigma_true, atol=1e-4)


def test_implied_volatility_nan_if_no_solution():
    # Give a nonsense "market price" too high for any sigma
    bad_price = 100.0
    sigma = implied_volatility(bad_price, 100, 100, 1.0, 0.05, "call")
    assert np.isnan(sigma)
