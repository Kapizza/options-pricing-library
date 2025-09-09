import numpy as np
from scipy.stats import norm
from .black_scholes import bs_d1_d2

def delta(S, K, T, r, sigma, option_type="call"):
    d = bs_d1_d2(S, K, T, r, sigma)
    d1 = d["d1"]
    if option_type.lower() == "call":
        return norm.cdf(d1)
    elif option_type.lower() == "put":
        return norm.cdf(d1) - 1.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(S, K, T, r, sigma):
    d = bs_d1_d2(S, K, T, r, sigma)
    d1 = d["d1"]
    sqrtT = np.sqrt(max(float(T), 1e-12))
    sigma = max(float(sigma), 1e-12)
    S = max(float(S), 1e-12)
    return norm.pdf(d1) / (S * sigma * sqrtT)

def vega(S, K, T, r, sigma):
    d = bs_d1_d2(S, K, T, r, sigma)
    d1 = d["d1"]
    sqrtT = np.sqrt(max(float(T), 1e-12))
    return S * norm.pdf(d1) * sqrtT  # per 1.00 of vol (not per 1%)

def theta(S, K, T, r, sigma, option_type="call"):
    d = bs_d1_d2(S, K, T, r, sigma)
    d1, d2 = d["d1"], d["d2"]
    sqrtT = np.sqrt(max(float(T), 1e-12))
    term1 = -(S * norm.pdf(d1) * sigma) / (2.0 * sqrtT)
    if option_type.lower() == "call":
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def rho(S, K, T, r, sigma, option_type="call"):
    d = bs_d1_d2(S, K, T, r, sigma)
    d2 = d["d2"]
    if option_type.lower() == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def vanna_volga(S, K, T, r, sigma):
    """
    Higher-order greeks (Black–Scholes):
      vega  = S * φ(d1) * √T
      volga = ∂²V/∂σ² = vega * d1 * d2 / σ
      vanna = ∂²V/(∂S∂σ) = (vega / S) * (1 - d1 / (σ √T))
    Returns dict: {"vanna": ..., "volga": ...}
    """
    d = bs_d1_d2(S, K, T, r, sigma)
    d1, d2 = d["d1"], d["d2"]
    S = max(float(S), 1e-12)
    sigma = max(float(sigma), 1e-12)
    sqrtT = np.sqrt(max(float(T), 1e-12))

    vega_bs = S * norm.pdf(d1) * sqrtT
    volga = vega_bs * d1 * d2 / sigma
    vanna = (vega_bs / S) * (1.0 - d1 / (sigma * sqrtT))
    return {"vanna": vanna, "volga": volga}
