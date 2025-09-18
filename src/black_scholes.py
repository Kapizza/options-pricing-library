import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import math

# --------------------------------------------------------------------------------------
# Core Black–Scholes
# --------------------------------------------------------------------------------------

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black–Scholes price for European calls/puts on a non-dividend-paying asset.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike
    T : float
        Time to maturity (in years)
    r : float
        Continuously compounded risk-free rate
    sigma : float
        Volatility (annualized)
    option_type : str
        "call" or "put"

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        intrinsic = S - K if option_type.lower() == "call" else K - S
        return float(max(intrinsic, 0.0))

    d = bs_d1_d2(S, K, T, r, sigma)
    d1, d2 = d["d1"], d["d2"]
    DF = math.exp(-r * T)

    if option_type.lower() == "call":
        return float(S * norm.cdf(d1) - K * DF * norm.cdf(d2))
    elif option_type.lower() == "put":
        return float(K * DF * norm.cdf(-d2) - S * norm.cdf(-d1))
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

# --------------------------------------------------------------------------------------
# Forward-measure helpers (useful for SABR and forward-based workflows)
# --------------------------------------------------------------------------------------

def bs_call_forward(F, K, T, iv):
    """
    Black–Scholes call price in *forward measure* (undiscounted).
    Returns E^{Q^T}[(S_T - K)^+] when F = E^{Q^T}[S_T].
    Final discounted call price = DF * bs_call_forward(F, K, T, iv) with DF = e^{-rT}.
    """
    if T <= 0:
        return max(F - K, 0.0)
    if iv <= 0 or K <= 0 or F <= 0:
        return max(F - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * iv * iv * T) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_put_forward(F, K, T, iv):
    """
    Black–Scholes put price in *forward measure* (undiscounted).
    Using parity in forward measure: P = C - (F - K).
    """
    return float(bs_call_forward(F, K, T, iv) - (F - K))


def black_scholes_price_from_forward(F, DF, K, T, iv, option_type="call"):
    """
    Convenience wrapper: BS price when you already have forward F and discount factor DF.
    """
    undisc = bs_call_forward(F, K, T, iv) if option_type.lower() == "call" else bs_put_forward(F, K, T, iv)
    return float(DF * undisc)

# --------------------------------------------------------------------------------------
# Implied volatility (robust) — Brent bracket + Newton polish
# --------------------------------------------------------------------------------------

def implied_vol_from_price(S, K, T, r, price, option_type="call", tol=1e-8):
    """
    Black–Scholes implied volatility from a given price.
    Robust: tries Brent in a wide bracket [1e-6, 5.0]; falls back to Newton if not bracketed.

    Parameters
    ----------
    S, K, T, r : floats
    price : float
        Observed option price
    option_type : "call" | "put"
    tol : float
        Solver tolerances

    Returns
    -------
    float
        Implied volatility (annualized). np.nan if not solvable.
    """
    S = float(S); K = float(K); T = float(T); r = float(r); price = float(price)

    # Guard rails
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return np.nan

    DF = math.exp(-r * T)
    discK = K * DF
    intrinsic = (S - discK) if option_type.lower() == "call" else (discK - S)
    intrinsic = max(0.0, intrinsic)

    # If below intrinsic, nudge up a hair
    if price < intrinsic:
        price = intrinsic + 1e-12

    # Root function for Brent
    def f(sig):
        return black_scholes_price(S, K, T, r, sig, option_type=option_type) - price

    lo, hi = 1e-6, 5.0
    flo, fhi = f(lo), f(hi)

    # Brent if bracketed
    if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
        try:
            return float(brentq(f, lo, hi, xtol=tol, rtol=tol, maxiter=200))
        except Exception:
            pass  # fall through to Newton

    # Newton fallback
    iv = 0.2
    for _ in range(20):
        d = bs_d1_d2(S, K, T, r, iv)
        d1 = d["d1"]
        # Vega in spot measure
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12:
            break
        px = black_scholes_price(S, K, T, r, iv, option_type=option_type)
        iv -= (px - price) / vega
        if iv <= 1e-6: iv = 1e-6
        if iv >= 5.0:  iv = 5.0
        if abs(px - price) < max(1e-10, tol * price):
            break

    return float(iv)
