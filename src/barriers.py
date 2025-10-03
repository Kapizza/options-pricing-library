"""
Barrier option pricing utilities (continuous monitoring) using Monte Carlo
with Brownian-bridge crossing detection, plus cash and asset digitals
under Black–Scholes with dividend yield q.

This version depends only on your `black_scholes_price` function from
`src/black_scholes.py` and avoids numpy.typing.

Design goals
- Minimal, dependency-light, vectorized numpy implementation
- Works consistently with your vanilla pricer for in–out parity
- Numerical guards for immediate knock-out and empty masks

References
- Broadie, Glasserman, Kou (1997): A continuity correction for discrete
  barrier options.
- Glasserman (2003): Monte Carlo Methods in Financial Engineering.

Notes
- Knock-in price is computed by parity: price_in = vanilla - price_out
- Rebate is a fixed cash amount paid at the detected hit time
- Antithetic variates are available for variance reduction
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

# Import exactly your vanilla pricer
from .black_scholes import black_scholes_price

BarrierType = Literal[
    "up-and-out",
    "down-and-out",
    "up-and-in",
    "down-and-in",
]

OptionType = Literal["call", "put"]


@dataclass
class MCConfig:
    n_paths: int = 200_000
    n_steps: int = 252
    antithetic: bool = True
    seed: Optional[int] = None
    dtype: type = np.float64


def _ensure_float(x: float) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("Inputs must be finite numbers")
    return x


def _normal_cdf(x):
    """Fast normal CDF approximation without SciPy and without np.erf.
    Hart (1968) style rational approximation on |x|, max abs error ~7.5e-8.
    Works on scalars and ndarrays.
    """
    z = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1
    one_over_sqrt2pi = 0.3989422804014327
    phi = one_over_sqrt2pi * np.exp(-0.5 * z * z)
    cdf_pos = 1.0 - phi * poly * t
    cdf = np.where(z >= 0.0, cdf_pos, 1.0 - cdf_pos)
    # Return scalar if input was scalar
    return float(cdf) if np.isscalar(x) else cdf


def _bridge_cross_up(x0: np.ndarray, x1: np.ndarray, b: float, sigma2_dt: float) -> np.ndarray:
    """Probability of crossing an up-barrier at log level b between x0 and x1.
    Uses the classic Brownian-bridge formula: exp(-2 (b - x0)(b - x1) / (sigma^2 dt))
    applied only when both endpoints are below the barrier. If either endpoint is
    already above the barrier, treat as certain hit for that interval.
    """
    below = (x0 < b) & (x1 < b)
    num = -2.0 * (b - x0) * (b - x1)
    p = np.exp(num / max(1e-32, sigma2_dt))
    out = np.zeros_like(p)
    out[below] = p[below]
    out[~below] = (x0[~below] >= b) | (x1[~below] >= b)
    return np.clip(out, 0.0, 1.0)


def _bridge_cross_down(x0: np.ndarray, x1: np.ndarray, b: float, sigma2_dt: float) -> np.ndarray:
    """Probability of crossing a down-barrier at log level b between x0 and x1."""
    above = (x0 > b) & (x1 > b)
    num = -2.0 * (x0 - b) * (x1 - b)
    p = np.exp(num / max(1e-32, sigma2_dt))
    out = np.zeros_like(p)
    out[above] = p[above]
    out[~above] = (x0[~above] <= b) | (x1[~above] <= b)
    return np.clip(out, 0.0, 1.0)


def _simulate_paths_log(S0: float, T: float, r: float, q: float, sigma: float, cfg: MCConfig):
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_paths
    m = cfg.n_steps
    dt = T / m
    mu = r - q
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    if cfg.antithetic:
        half = (n + 1) // 2
        Z = rng.standard_normal((half, m)).astype(cfg.dtype, copy=False)
        Z = np.vstack([Z, -Z])[:n]
    else:
        Z = rng.standard_normal((n, m)).astype(cfg.dtype, copy=False)

    X = np.empty((n, m + 1), dtype=cfg.dtype)
    X[:, 0] = math.log(S0)
    for t in range(m):
        X[:, t + 1] = X[:, t] + drift + vol * Z[:, t]
    return X, dt, sigma * sigma * dt


def _discount(x, r: float, t):
    return np.asarray(x) * np.exp(-r * np.asarray(t))


def barrier_price_mc(
    S0: float,
    K: float,
    H: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option: OptionType = "call",
    barrier: BarrierType = "up-and-out",
    rebate: float = 0.0,
    cfg: Optional[MCConfig] = None,
) -> float:
    """
    Continuous-monitoring barrier option price via Monte Carlo with Brownian bridge.

    Parameters
    S0, K, H, T, r, q, sigma : floats
    option : "call" or "put"
    barrier : one of {"up-and-out","down-and-out","up-and-in","down-and-in"}
    rebate : cash rebate paid at the hitting time (0 for none)
    cfg : MCConfig

    Returns
    Price as float.

    Notes
    - For knock-in, uses in–out parity with your `black_scholes_price`.
    - Falls back to MC vanilla only if necessary (should not be needed here).
    """
    S0 = _ensure_float(S0)
    K = _ensure_float(K)
    H = _ensure_float(H)
    T = max(1e-10, _ensure_float(T))
    r = _ensure_float(r)
    q = _ensure_float(q)
    sigma = max(1e-12, _ensure_float(sigma))

    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")

    if cfg is None:
        cfg = MCConfig()

    # Immediate knock-out at t=0
    if barrier == "up-and-out" and S0 >= H:
        return float(rebate)
    if barrier == "down-and-out" and S0 <= H:
        return float(rebate)

    X, dt, sigma2_dt = _simulate_paths_log(S0, T, r, q, sigma, cfg)
    S = np.exp(X)

    logH = math.log(H)
    x0 = X[:, :-1]
    x1 = X[:, 1:]

    if barrier in ("up-and-out", "up-and-in"):
        p_cross = _bridge_cross_up(x0, x1, logH, sigma2_dt)
    else:
        p_cross = _bridge_cross_down(x0, x1, logH, sigma2_dt)

    rng = np.random.default_rng(cfg.seed if cfg.seed is None else cfg.seed + 7)
    U = rng.random(p_cross.shape)
    hit_matrix = (U < p_cross)

    # First hit information
    hit_any = hit_matrix.any(axis=1)
    first_hit_idx = np.where(hit_any, hit_matrix.argmax(axis=1), -1)
    first_hit_time = np.where(hit_any, (first_hit_idx + 1) * dt, np.nan)

    ST = S[:, -1]
    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    is_out = barrier in ("up-and-out", "down-and-out")

    if is_out:
        alive_mask = ~hit_any
        if np.any(alive_mask):
            discounted_payoff = _discount(payoff[alive_mask], r, T)
            price_out = float(np.mean(discounted_payoff))
        else:
            price_out = 0.0
        if rebate > 0.0 and np.any(hit_any):
            disc_rebate = _discount(rebate, r, first_hit_time[hit_any])
            price_out += float(np.mean(disc_rebate))
        return price_out

    # Knock-in via parity
    vanilla = black_scholes_price(S0, K, T, r, sigma, option_type=option, q=q)

    alive_mask = ~hit_any
    if np.any(alive_mask):
        discounted_payoff_out = _discount(payoff[alive_mask], r, T)
        price_out = float(np.mean(discounted_payoff_out))
    else:
        price_out = 0.0
    if rebate > 0.0 and np.any(hit_any):
        disc_rebate = _discount(rebate, r, first_hit_time[hit_any])
        price_out += float(np.mean(disc_rebate))

    price_in = float(vanilla - price_out)
    return price_in


# -----------------------------
# Digitals under Black–Scholes
# -----------------------------

def digital_cash_bsm(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option: OptionType = "call", cash: float = 1.0) -> float:
    """Cash-or-nothing digital in BSM with dividend yield q.
    Price = cash * e^{-rT} * N(d2) for call, N(-d2) for put.
    """
    S = _ensure_float(S)
    K = _ensure_float(K)
    T = max(1e-10, _ensure_float(T))
    r = _ensure_float(r)
    q = _ensure_float(q)
    sigma = max(1e-12, _ensure_float(sigma))

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option == "call":
        return float(cash * math.exp(-r * T) * _normal_cdf(d2))
    else:
        return float(cash * math.exp(-r * T) * _normal_cdf(-d2))


def digital_asset_bsm(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option: OptionType = "call") -> float:
    """Asset-or-nothing digital in BSM with dividend yield q.
    Price = S e^{-qT} N(d1) for call, S e^{-qT} N(-d1) for put.
    """
    S = _ensure_float(S)
    K = _ensure_float(K)
    T = max(1e-10, _ensure_float(T))
    r = _ensure_float(r)
    q = _ensure_float(q)
    sigma = max(1e-12, _ensure_float(sigma))

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if option == "call":
        return float(S * math.exp(-q * T) * _normal_cdf(d1))
    else:
        return float(S * math.exp(-q * T) * _normal_cdf(-d1))


__all__ = [
    "MCConfig",
    "barrier_price_mc",
    "digital_cash_bsm",
    "digital_asset_bsm",
]
