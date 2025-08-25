# options_pricing/strategies.py

import numpy as np
from .black_scholes import black_scholes_price


# --------------------
# Single Option Legs
# --------------------
def long_call(S, K, T, r, sigma):
    """Price of a long call option."""
    return black_scholes_price(S, K, T, r, sigma, option_type="call")

def short_call(S, K, T, r, sigma):
    """Price of a short call option (negative of long call)."""
    return -black_scholes_price(S, K, T, r, sigma, option_type="call")

def long_put(S, K, T, r, sigma):
    """Price of a long put option."""
    return black_scholes_price(S, K, T, r, sigma, option_type="put")

def short_put(S, K, T, r, sigma):
    """Price of a short put option (negative of long put)."""
    return -black_scholes_price(S, K, T, r, sigma, option_type="put")


# --------------------
# Spreads
# --------------------
def bull_call_spread(S, K1, K2, T, r, sigma):
    """
    Bull Call Spread:
    Long call at K1 (lower strike), Short call at K2 (higher strike).
    """
    return long_call(S, K1, T, r, sigma) + short_call(S, K2, T, r, sigma)

def bear_put_spread(S, K1, K2, T, r, sigma):
    """
    Bear Put Spread:
    Long put at K1 (higher strike), Short put at K2 (lower strike).
    """
    return long_put(S, K1, T, r, sigma) + short_put(S, K2, T, r, sigma)


def butterfly_spread(S, K1, K2, K3, r, T, sigma):
    """
    Computes the price of a butterfly spread using European calls.

    Parameters:
    S : float
        Current stock price
    K1 : float
        Lower strike price
    K2 : float
        Middle strike price (usually the target price)
    K3 : float
        Upper strike price
    r : float
        Risk-free interest rate (annual)
    T : float
        Time to maturity in years
    sigma : float
        Volatility of the underlying asset (annual)

    Returns:
    float
        Price of the butterfly spread
    """
    # Long 1 call at K1
    c1 = black_scholes_price(S, K1, T, r, sigma, option_type="call")
    # Short 2 calls at K2
    c2 = 2 * black_scholes_price(S, K2, T, r, sigma, option_type="call")
    # Long 1 call at K3
    c3 = black_scholes_price(S, K3, T, r, sigma, option_type="call")

    # Butterfly spread price
    butterfly_price = c1 - c2 + c3
    return butterfly_price  



# --------------------
# Combinations
# --------------------
def straddle(S, K, T, r, sigma):
    """
    Straddle:
    Long call + Long put at the same strike K.
    """
    return long_call(S, K, T, r, sigma) + long_put(S, K, T, r, sigma)

def strangle(S, K1, K2, T, r, sigma):
    """
    Strangle:
    Long put at K1 (lower strike), Long call at K2 (higher strike).
    """
    return long_put(S, K1, T, r, sigma) + long_call(S, K2, T, r, sigma)

def collar(S, K_put, K_call, T, r, sigma):
    """
    Collar:
    Long put at K_put, Short call at K_call.
    """
    return long_put(S, K_put, T, r, sigma) + short_call(S, K_call, T, r, sigma)


# --------------------
# Payoff Diagram Utility
# --------------------
def payoff_diagram(strategy_fn, S_range, **kwargs):
    """
    Compute payoff diagram for a given strategy.
    
    Parameters:
    - strategy_fn: function (one of the strategy functions above)
    - S_range: array of stock prices
    - kwargs: arguments for the strategy function (K, T, r, sigma, etc.)
    
    Returns:
    - S_range, payoffs
    """
    payoffs = np.array([strategy_fn(S, **kwargs) for S in S_range])
    return S_range, payoffs



