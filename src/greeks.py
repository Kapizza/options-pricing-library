# src/greeks.py
import numpy as np
from scipy.stats import norm
from .black_scholes import bs_d1_d2 as _bs_d1_d2


# ---------- internal helpers ----------

def _as_array(x):
    """Ensure numpy array view (no copy for ndarray inputs)."""
    return np.asarray(x, dtype=float)

def _d1d2(S, K, T, r, sigma, q=0.0):
    """
    Robust d1, d2 with dividend yield q.
    Vectorized over numpy arrays.
    """
    S = _as_array(S); K = _as_array(K); T = _as_array(T)
    r = _as_array(r); sigma = _as_array(sigma); q = _as_array(q)

    # floors to avoid division by zero / log invalids
    T_ = np.maximum(T, 1e-12)
    sig_ = np.maximum(sigma, 1e-12)
    S_ = np.maximum(S, 1e-300)  # keep log well-defined
    d = _bs_d1_d2(S_, K, T_, r, sig_, q=q)
    d1 = _as_array(d["d1"])
    d2 = _as_array(d["d2"])
    return d1, d2

# ---------- Greeks with dividend yield q (default q=0.0 for backward compatibility) ----------

def delta(S, K, T, r, sigma, option_type="call", *, q=0.0):
    """
    Black–Scholes delta with continuous dividend yield q.
    Call:  Δ = e^{-qT} N(d1)
    Put :  Δ = e^{-qT} [N(d1) - 1]
    """
    d1, _ = _d1d2(S, K, T, r, sigma, q=q)
    w = np.exp(-np.maximum(_as_array(q) * _as_array(T), 0.0))
    Nd1 = norm.cdf(d1)
    if option_type.lower() == "call":
        return w * Nd1
    elif option_type.lower() == "put":
        return w * (Nd1 - 1.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma, *, q=0.0):
    """
    Γ = e^{-qT} φ(d1) / (S σ √T)
    """
    d1, _ = _d1d2(S, K, T, r, sigma, q=q)
    T_ = np.maximum(_as_array(T), 1e-12)
    sig_ = np.maximum(_as_array(sigma), 1e-12)
    S_ = np.maximum(_as_array(S), 1e-300)
    w = np.exp(-np.maximum(_as_array(q) * T_, 0.0))
    return w * norm.pdf(d1) / (S_ * sig_ * np.sqrt(T_))


def vega(S, K, T, r, sigma, *, q=0.0):
    """
    Vega per unit (1.00) of volatility:
    V = e^{-qT} S φ(d1) √T
    (If you want per 1% vol, divide by 100.)
    """
    d1, _ = _d1d2(S, K, T, r, sigma, q=q)
    T_ = np.maximum(_as_array(T), 1e-12)
    w = np.exp(-np.maximum(_as_array(q) * T_, 0.0))
    return w * _as_array(S) * norm.pdf(d1) * np.sqrt(T_)


def theta(S, K, T, r, sigma, option_type="call", *, q=0.0):
    """
    Calendar-year theta with dividend yield q (Merton):
      Call: θ = - e^{-qT} S φ(d1) σ/(2√T) + q e^{-qT} S N(d1) - r K e^{-rT} N(d2)
      Put : θ = - e^{-qT} S φ(d1) σ/(2√T) - q e^{-qT} S N(-d1) + r K e^{-rT} N(-d2)
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q=q)
    T_ = np.maximum(_as_array(T), 1e-12)
    sig_ = np.maximum(_as_array(sigma), 1e-12)
    S_ = _as_array(S); K_ = _as_array(K); r_ = _as_array(r); q_ = _as_array(q)

    disc_r = np.exp(-r_ * T_)
    disc_q = np.exp(-q_ * T_)
    term1 = -disc_q * S_ * norm.pdf(d1) * sig_ / (2.0 * np.sqrt(T_))

    if option_type.lower() == "call":
        return term1 + q_ * disc_q * S_ * norm.cdf(d1) - r_ * K_ * disc_r * norm.cdf(d2)
    elif option_type.lower() == "put":
        return term1 - q_ * disc_q * S_ * norm.cdf(-d1) + r_ * K_ * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, option_type="call", *, q=0.0):
    """
    Rho (per 1.00 change in r):
      Call:  ρ =  K T e^{-rT} N(d2)
      Put :  ρ = -K T e^{-rT} N(-d2)
    (Note: unaffected directly by q in the standard formulation.)
    """
    _, d2 = _d1d2(S, K, T, r, sigma, q=q)
    K_ = _as_array(K); T_ = _as_array(T); r_ = _as_array(r)
    disc_r = np.exp(-r_ * T_)
    if option_type.lower() == "call":
        return K_ * T_ * disc_r * norm.cdf(d2)
    elif option_type.lower() == "put":
        return -K_ * T_ * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def vanna_volga(S, K, T, r, sigma, *, q=0.0):
    """
    Higher-order greeks under continuous dividend yield q:

      vega  = e^{-qT} S φ(d1) √T
      volga = ∂²V/∂σ² = vega * d1 * d2 / σ
      vanna = ∂²V/(∂S∂σ) = (vega / S) * (1 - d1 / (σ √T))

    Returns dict: {"vanna": vanna, "volga": volga}
    """
    d1, d2 = _d1d2(S, K, T, r, sigma, q=q)
    T_ = np.maximum(_as_array(T), 1e-12)
    sig_ = np.maximum(_as_array(sigma), 1e-12)
    S_ = np.maximum(_as_array(S), 1e-300)

    vega_full = vega(S_, K, T_, r, sig_, q=q)
    volga = vega_full * d1 * d2 / sig_
    vanna = (vega_full / S_) * (1.0 - d1 / (sig_ * np.sqrt(T_)))
    return {"vanna": vanna, "volga": volga}
