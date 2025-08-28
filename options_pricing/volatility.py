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
    """
    if price <= 0:
        raise ValueError("Option price must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity must be positive.")

    intrinsic = max(0.0, S - K if option_type == "call" else K - S)

    if option_type == "call":
        max_price = S
    else:
        max_price = K * np.exp(-r * T)

    if price <= intrinsic + 1e-8 or price >= max_price - 1e-8:
        return np.nan

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price

    vol_lower, vol_upper = 1e-9, 10.0

    try:
        return brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iterations)
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
    """
    nT, nK = len(maturities), len(strikes)
    surface = np.full((nT, nK), np.nan)

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            surface[i, j] = implied_volatility(
                prices[i, j], S, K, T, r, option_type
            )

    return surface


def moneyness_grid(
    strikes: np.ndarray,
    maturities: np.ndarray,
    S: float,
    r: float,
    kind: str = "K_over_F",
) -> np.ndarray:
    """
    Compute a moneyness grid aligned with an IV surface.

    Parameters
    ----------
    kind : {"K_over_F", "K_over_S", "log"}
        - "K_over_F": K / (S * exp(rT))   [market-standard forward moneyness]
        - "K_over_S": K / S
        - "log":      ln(S/K)             [theoretical log-moneyness]
    """
    strikes = np.asarray(strikes)
    maturities = np.asarray(maturities)

    Kmesh, Tmesh = np.meshgrid(strikes, maturities)

    if kind == "K_over_S":
        return Kmesh / float(S)
    elif kind == "K_over_F":
        F = S * np.exp(r * maturities)
        return Kmesh / F[:, None]
    elif kind == "log":
        return np.log(float(S) / Kmesh)
    else:
        raise ValueError("kind must be one of {'K_over_F','K_over_S','log'}")
