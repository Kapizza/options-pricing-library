import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import math

# --------------------------------------------------------------------------------------
# Core Black–Scholes (Merton model with continuous dividend yield q)
# --------------------------------------------------------------------------------------

def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black–Scholes–Merton price for European calls/puts with continuous dividend yield q.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (cont. comp.)
    sigma : float
        Volatility (annualized)
    option_type : str
        "call" or "put"
    q : float, default 0.0
        Continuous dividend yield

    Returns
    -------
    float
        Option price
    """
    if T <= 0:
        intrinsic = (S - K) if option_type.lower() == "call" else (K - S)
        return float(max(intrinsic, 0.0))

    d = bs_d1_d2(S, K, T, r, sigma, q=q)
    d1, d2 = d["d1"], d["d2"]
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type.lower() == "call":
        return float(disc_q * S * norm.cdf(d1) - K * disc_r * norm.cdf(d2))
    elif option_type.lower() == "put":
        return float(K * disc_r * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_d1_d2(S, K, T, r, sigma, q=0.0):
    """
    Vectorized d1 and d2 for Black–Scholes–Merton with dividend yield q.
    Accepts scalars or numpy arrays (broadcasted).
    Returns dict: {"d1": ndarray, "d2": ndarray}
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    eps = 1e-12
    S_ = np.maximum(S, eps)
    K_ = np.maximum(K, eps)
    T_ = np.maximum(T, eps)
    sig_ = np.maximum(sigma, eps)

    sqrtT = np.sqrt(T_)
    d1 = (np.log(S_ / K_) + (r - q + 0.5 * sig_ * sig_) * T_) / (sig_ * sqrtT)
    d2 = d1 - sig_ * sqrtT
    return {"d1": d1, "d2": d2}


# --------------------------------------------------------------------------------------
# Forward-measure helpers (compatible with q)
# --------------------------------------------------------------------------------------

def bs_call_forward(F, K, T, iv):
    """Forward-measure Black–Scholes call (undiscounted)."""
    if T <= 0:
        return max(F - K, 0.0)
    if iv <= 0 or K <= 0 or F <= 0:
        return max(F - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * iv * iv * T) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


def bs_put_forward(F, K, T, iv):
    """Forward-measure put (undiscounted)."""
    return float(bs_call_forward(F, K, T, iv) - (F - K))


def black_scholes_price_from_forward(F, DF, K, T, iv, option_type="call"):
    """Price when you already have forward F and discount factor DF."""
    undisc = bs_call_forward(F, K, T, iv) if option_type.lower() == "call" else bs_put_forward(F, K, T, iv)
    return float(DF * undisc)

# --------------------------------------------------------------------------------------
# Implied volatility solver (with q support)
# --------------------------------------------------------------------------------------

def implied_vol_from_price(S, K, T, r, price, option_type="call", q=0.0, tol=1e-8):
    """
    Implied volatility from price using Black–Scholes–Merton with q.
    Uses Brent root-finding + Newton fallback.
    """
    S, K, T, r, price = map(float, (S, K, T, r, price))

    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return np.nan

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    F = S * disc_q / disc_r

    intrinsic = (S - K) if option_type.lower() == "call" else (K - S)
    intrinsic = max(0.0, intrinsic)
    if price < intrinsic:
        price = intrinsic + 1e-12

    def f(sig):
        return black_scholes_price(S, K, T, r, sig, option_type=option_type, q=q) - price

    lo, hi = 1e-6, 5.0
    flo, fhi = f(lo), f(hi)

    if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
        try:
            return float(brentq(f, lo, hi, xtol=tol, rtol=tol, maxiter=200))
        except Exception:
            pass

    iv = 0.2
    for _ in range(20):
        d = bs_d1_d2(S, K, T, r, iv, q=q)
        d1 = d["d1"]
        vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12:
            break
        px = black_scholes_price(S, K, T, r, iv, option_type=option_type, q=q)
        iv -= (px - price) / vega
        iv = min(max(iv, 1e-6), 5.0)
        if abs(px - price) < max(1e-10, tol * price):
            break

    return float(iv)

# --------------------------------------------------------------------------------------
# Delta helper (kept for compatibility with your notebooks)
# --------------------------------------------------------------------------------------

def bs_delta_call(S, K, T, r, q, sigma):
    """Black–Scholes–Merton call delta with dividend yield q."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1_d2(S, K, T, r, sigma, q=q)["d1"]
    return math.exp(-q * T) * norm.cdf(d1)

def iv_from_surface(surf, S, K, T, r, q):
    """Query IV from SVI surface using log-moneyness k=ln(K/F_T)."""
    if T <= 0:
        return np.nan
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)
    return surf.iv(np.array([k]), T).item()
