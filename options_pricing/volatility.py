# options_pricing/volatility.py

import numpy as np
from scipy.optimize import brentq
from .black_scholes import black_scholes_price

def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iterations: int = 100
) -> float:
    """
    Compute implied volatility using Brent's method with stability guards.

    Parameters
    ----------
    price : float
        Observed market option price
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    option_type : str
        "call" or "put"
    tol : float
        Root finding tolerance
    max_iterations : int
        Max iterations for root solver

    Returns
    -------
    float
        Implied volatility, or NaN if no valid solution
    """

    if price <= 0:
        raise ValueError("Option price must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity must be positive.")

    # Intrinsic value (lower bound of option price)
    intrinsic = max(0.0, S - K if option_type == "call" else K - S)

    # Rough upper bound (for calls it's S, for puts it's K*exp(-rT) + maybe S)
    if option_type == "call":
        max_price = S
    else:
        max_price = K * np.exp(-r * T)

    # If market price is outside no-arbitrage bounds → no solution
    if price <= intrinsic + 1e-8 or price >= max_price - 1e-8:
        return np.nan

    # Function for root finding
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price

    vol_lower, vol_upper = 1e-9, 10.0  # extend range for robustness

    try:
        implied_vol = brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iterations)
        return implied_vol
    except ValueError:
        return np.nan


def implied_vol_surface(
    prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float,
    option_type: str = "call"
) -> np.ndarray:
    """
    Build implied volatility surface (strike × maturity grid).

    Parameters
    ----------
    prices : np.ndarray
        Matrix of option prices with shape (len(maturities), len(strikes))
    S : float
        Spot price
    strikes : np.ndarray
        Array of strike prices
    maturities : np.ndarray
        Array of maturities (in years)
    r : float
        Risk-free rate
    option_type : str
        "call" or "put"

    Returns
    -------
    np.ndarray
        Implied volatility surface of shape (len(maturities), len(strikes))
    """
    nT, nK = len(maturities), len(strikes)
    surface = np.full((nT, nK), np.nan)  # fill with NaN initially

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            surface[i, j] = implied_volatility(
                prices[i, j], S, K, T, r, option_type
            )

    return surface
