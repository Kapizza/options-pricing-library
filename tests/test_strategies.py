# tests/test_strategies.py

import numpy as np
import pytest

from src.strategies import (
    long_call, short_call, long_put, short_put,
    bull_call_spread, bear_put_spread,
    straddle, strangle, collar, payoff_diagram, butterfly_spread
)
from src.black_scholes import black_scholes_price

# Common test parameters
S = 100   # spot price
K = 100   # strike
K1, K2 = 95, 105
T = 1.0   # 1 year
r = 0.05  # risk-free rate
sigma = 0.2  # volatility


def test_long_short_symmetry():
    """Check that short = -long for calls and puts."""
    assert np.isclose(short_call(S, K, T, r, sigma), -long_call(S, K, T, r, sigma))
    assert np.isclose(short_put(S, K, T, r, sigma), -long_put(S, K, T, r, sigma))


def test_bull_call_spread_equivalence():
    """Bull call spread should equal long_call(K1) + short_call(K2)."""
    spread_price = bull_call_spread(S, K1, K2, T, r, sigma)
    manual_price = long_call(S, K1, T, r, sigma) + short_call(S, K2, T, r, sigma)
    assert np.isclose(spread_price, manual_price)


def test_bear_put_spread_equivalence():
    """Bear put spread should equal long_put(K2) + short_put(K1)."""
    spread_price = bear_put_spread(S, K2, K1, T, r, sigma)
    manual_price = long_put(S, K2, T, r, sigma) + short_put(S, K1, T, r, sigma)
    assert np.isclose(spread_price, manual_price)


def test_straddle_equivalence():
    """Straddle should equal long_call + long_put at same strike."""
    strat_price = straddle(S, K, T, r, sigma)
    manual_price = long_call(S, K, T, r, sigma) + long_put(S, K, T, r, sigma)
    assert np.isclose(strat_price, manual_price)


def test_strangle_equivalence():
    """Strangle = long_put(K1) + long_call(K2)."""
    strat_price = strangle(S, K1, K2, T, r, sigma)
    manual_price = long_put(S, K1, T, r, sigma) + long_call(S, K2, T, r, sigma)
    assert np.isclose(strat_price, manual_price)


def test_collar_equivalence():
    """Collar = long_put(K1) + short_call(K2)."""
    strat_price = collar(S, K1, K2, T, r, sigma)
    manual_price = long_put(S, K1, T, r, sigma) + short_call(S, K2, T, r, sigma)
    assert np.isclose(strat_price, manual_price)


def test_payoff_diagram_runs():
    """Payoff diagram should return arrays of same length."""
    S_range = np.linspace(50, 150, 10)
    S_vals, payoffs = payoff_diagram(straddle, S_range, K=K, T=T, r=r, sigma=sigma)
    assert len(S_vals) == len(payoffs)
    assert isinstance(payoffs, np.ndarray)


# --------------------
# Numerical sanity checks
# --------------------

def test_positive_prices():
    """Long calls and puts should have non-negative price."""
    assert long_call(S, K, T, r, sigma) >= 0
    assert long_put(S, K, T, r, sigma) >= 0


def test_spreads_are_cheaper_than_single_options():
    """Spreads should cost less than outright long option."""
    call_spread = bull_call_spread(S, K1, K2, T, r, sigma)
    put_spread = bear_put_spread(S, K2, K1, T, r, sigma)

    assert call_spread < long_call(S, K1, T, r, sigma)
    assert put_spread < long_put(S, K2, T, r, sigma)


def test_straddle_more_expensive_than_single_option():
    """Straddle should be more expensive than either a single call or put."""
    price = straddle(S, K, T, r, sigma)
    call_price = long_call(S, K, T, r, sigma)
    put_price = long_put(S, K, T, r, sigma)
    assert price > call_price
    assert price > put_price


def test_strangle_cheaper_than_straddle():
    """Strangle is usually cheaper than straddle (since strikes are OTM)."""
    straddle_price = straddle(S, K, T, r, sigma)
    strangle_price = strangle(S, K1, K2, T, r, sigma)
    assert strangle_price < straddle_price


def test_butterfly_spread_value():
    S = 100
    K1 = 95
    K2 = 100
    K3 = 105
    r = 0.05
    T = 0.5
    sigma = 0.2

    price = butterfly_spread(S, K1, K2, K3, r, T, sigma)

    # Use the same call_price function for expected value
    expected_price = black_scholes_price(S=S, K=K1, T=T, r=r, sigma=sigma, option_type="call") - 2 * black_scholes_price(S=S, K=K2, T=T, r=r, sigma=sigma, option_type="call") + black_scholes_price(S=S, K=K3, T=T, r=r, sigma=sigma, option_type="call")


    assert abs(price - expected_price) < 1e-12


def test_butterfly_spread_non_negative():
    # Butterfly spreads generally have non-negative prices for typical strikes
    S = 100
    K1 = 95
    K2 = 100
    K3 = 105
    r = 0.05
    T = 0.5
    sigma = 0.2

    price = butterfly_spread(S, K1, K2, K3, r, T, sigma)
    assert price >= 0