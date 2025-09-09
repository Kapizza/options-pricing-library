import numpy as np
from scipy.stats import norm
import math

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option pricing formula.
    """
    d = bs_d1_d2(S, K, T, r, sigma)
    d1, d2 = d["d1"], d["d2"]

    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def bs_d1_d2(S, K, T, r, sigma):
    """
    Compute d1 and d2 for Black–Scholes.
    Returns a dict: {"d1": ..., "d2": ...}
    """
    eps = 1e-12
    S = max(float(S), eps)
    K = max(float(K), eps)
    T = max(float(T), eps)
    sigma = max(float(sigma), eps)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return {"d1": d1, "d2": d2}
