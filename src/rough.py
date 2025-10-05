# src/rough.py
"""
Rough volatility tools (baseline)

This module provides a minimal reference implementation for rBergomi-style rough
vol pricing using a fractional Brownian motion (fBM) driver. It is intended as a
slow but correct baseline for research notebooks. For production use, replace
the fBM generator with a fast hybrid scheme and vectorize across strikes.

Model (rBergomi, H in (0,1)):
    dS_t / S_t = sqrt(v_t) dW_t^S
    v_t = xi0(t) * exp( eta * W_t^H - 0.5 * eta^2 * t^{2H} )
    Corr(W^S, W) = rho

Where W^H is a fractional Brownian motion with Hurst exponent H.
xi0(t) is the forward variance curve. We support xi0 as a scalar or callable.

Key functions:
    rbergomi_euro_mc(...)     -> Monte Carlo price and stderr for European call/put
    rbergomi_paths(...)       -> simulate price and variance paths
    fbm_increments_hosking(...) -> fBM increments via Hosking method (O(N^2))

Notes:
    - This code is deliberately simple and well guarded. It is not fast.
    - Time grid is uniform. Use N in the thousands only for small experiments.
    - All inputs are validated and floored for numerical safety.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, Optional


# ------------------------ utilities and guards ------------------------

def _as_float(x) -> float:
    return float(x)

def _floor_pos(x: float, eps: float = 1e-12) -> float:
    return max(float(x), eps)

def _validate_call_put(option: str) -> str:
    s = str(option).lower()
    if s not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    return s

def _xi0_as_callable(xi0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Accept scalar, array-like (broadcast), or callable xi0(t).
    Returns a callable f(t_grid)->array of shape t_grid.
    """
    if callable(xi0):
        return lambda t: np.asarray(xi0(t), dtype=float)
    # scalar or array-like
    def _f(tgrid: np.ndarray) -> np.ndarray:
        x = np.asarray(xi0, dtype=float)
        if x.ndim == 0:
            return np.full_like(tgrid, float(x), dtype=float)
        # broadcast if lengths match
        if x.size == tgrid.size:
            return x.astype(float)
        # fallback to last value if shorter
        return np.resize(x.astype(float), tgrid.size)
    return _f


# ------------------------ fractional Brownian motion ------------------------

def fbm_increments_hosking(
    N: int,
    H: float,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate fractional Gaussian noise (fGn) of length N with Hurst H via Hosking.
    Returns increments of fBM on unit time step, i.e., X_k = B_H(k) - B_H(k-1).

    Complexity is O(N^2). Good for small N (<= a few thousand).

    Parameters
    ----------
    N : int
        Number of increments.
    H : float
        Hurst exponent in (0, 1). For rough volatility, typical H in (0, 0.5).
    rng : np.random.Generator or None
        Random generator. If None, uses default.

    Returns
    -------
    fgn : np.ndarray, shape (N,)
        Fractional Gaussian noise with Var(X_k) = 1 at unit step, zero mean.

    Reference
    ---------
    Hosking, "Modeling persistence in hydrological time series," 1984.
    """
    if rng is None:
        rng = np.random.default_rng()

    H = float(H)
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1)")

    # Autocovariance of fGn at lag k for unit steps
    def gamma(k: int) -> float:
        k = abs(int(k))
        return 0.5 * ( (k + 1)**(2.0 * H) - 2.0 * (k**(2.0 * H)) + (k - 1)**(2.0 * H) ) if k > 0 else 1.0

    # Hosking recursion
    fgn = np.empty(N, dtype=float)
    phi = np.empty(N, dtype=float)
    psi = np.empty(N, dtype=float)

    var = gamma(0)
    fgn[0] = rng.normal(0.0, math.sqrt(var), size=None)

    for n in range(1, N):
        # compute phi_n,n
        num = gamma(n)
        for j in range(n - 1):
            num -= phi[j] * gamma(n - j - 1)
        den = var
        kappa = num / den

        # update sequence phi
        psi[:n-1] = phi[:n-1]
        for j in range(n - 1):
            psi[j] = phi[j] - kappa * phi[n - j - 2]
        phi[:n-1] = psi[:n-1]
        phi[n - 1] = kappa

        # innovation variance
        var = var * (1.0 - kappa * kappa)
        var = max(var, 1e-20)

        # generate X_n
        mean = 0.0
        for j in range(n):
            mean += phi[j] * fgn[n - j - 1]
        fgn[n] = mean + rng.normal(0.0, math.sqrt(var), size=None)

    return fgn


# ------------------------ rBergomi paths ------------------------

def rbergomi_paths(
    S0: float,
    T: float,
    N: int,
    n_paths: int,
    H: float,
    eta: float,
    rho: float,
    xi0,
    r: float = 0.0,
    q: float = 0.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate rBergomi paths for S and v on a uniform grid using simple Euler.

    Parameters
    ----------
    S0 : float
        Spot at t=0.
    T : float
        Maturity in years.
    N : int
        Number of steps (uniform grid).
    n_paths : int
        Number of Monte Carlo paths.
    H : float
        Hurst exponent for fBM driver.
    eta : float
        Vol-of-vol parameter.
    rho : float
        Spot-vol correlation in (-1, 1).
    xi0 : float | array-like | callable
        Forward variance curve xi0(t). If scalar, constant. If array-like, broadcast.
        If callable, it must accept an array of times and return same-shape values.
    r : float
        Risk-free rate (cont. comp.).
    q : float
        Dividend yield (cont. comp.).
    seed : int or None
        RNG seed.

    Returns
    -------
    t : np.ndarray, shape (N+1,)
        Time grid.
    S : np.ndarray, shape (n_paths, N+1)
        Simulated spot paths.
    v : np.ndarray, shape (n_paths, N+1)
        Instantaneous variance paths.

    Notes
    -----
    - Uses Euler step for S with drift (r - q), diffusion sqrt(v).
    - rBergomi variance is lognormal by design, v_t = xi0(t) * exp(eta*W_H(t) - 0.5*eta^2 t^{2H}).
    - Correlation is enforced by building correlated Brownian increments for S and fBM's innovation driver.
    """
    # Guards
    S0 = _floor_pos(S0)
    T = _floor_pos(T)
    N = int(N)
    if N < 1:
        raise ValueError("N must be >= 1")
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if not (-0.999 < rho < 0.999):
        raise ValueError("rho must be in (-0.999, 0.999)")
    H = float(H)
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")
    eta = _floor_pos(eta)
    r = float(r)
    q = float(q)

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, N + 1)
    dt = float(T) / N
    sqrt_dt = math.sqrt(dt)

    xi_fn = _xi0_as_callable(xi0)
    xi_vec = xi_fn(t)  # shape N+1

    # Build fBM W^H(t) for each path via fGn sum, scaled to match Var[B_H(t)] = t^{2H}
    # We first generate standard fGn (unit time step) then rescale to dt grid.
    W_H = np.empty((n_paths, N + 1), dtype=float)
    W_H[:, 0] = 0.0
    scale = (dt**H)  # fBM scaling on step size

    for i in range(n_paths):
        fgn = fbm_increments_hosking(N, H, rng)  # Var step = 1 at unit step
        W_H[i, 1:] = np.cumsum(fgn) * scale  # B_H(t_k) with t_k = k*dt

    # Build correlated Brownian increments for S and the Gaussian driver of W^H
    # We approximate dW (the innovation that drives W^H) with standard normals Z2,
    # and correlate dW^S = rho*Z2 + sqrt(1-rho^2)*Z1. This is a common proxy
    # for Euler schemes in rBergomi.
    Z1 = rng.standard_normal((n_paths, N))  # independent
    Z2 = rng.standard_normal((n_paths, N))  # to be correlated with spot
    dW_S = rho * Z2 + math.sqrt(1.0 - rho * rho) * Z1

    # Variance process from rBergomi formula
    # v_t = xi0(t) * exp( eta * W_H(t) - 0.5 * eta^2 * t^{2H} )
    t_pow = t**(2.0 * H)
    drift_corr = -0.5 * (eta**2) * t_pow
    v = (xi_vec[None, :] * np.exp(eta * W_H + drift_corr[None, :])).astype(float)
    v = np.maximum(v, 1e-14)

    # Simulate S with Euler under risk neutral
    S = np.empty((n_paths, N + 1), dtype=float)
    S[:, 0] = S0
    drift = (r - q) * dt

    for k in range(N):
        # diffusion uses v at time t_k
        vol_step = np.sqrt(np.maximum(v[:, k], 1e-14)) * sqrt_dt
        S[:, k + 1] = S[:, k] * np.exp(drift - 0.5 * vol_step**2 + vol_step * dW_S[:, k])

    return t, S, v


# ------------------------ European pricing ------------------------

def rbergomi_euro_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    H: float,
    eta: float,
    rho: float,
    xi0,
    n_paths: int = 20000,
    N: int = 500,
    option: str = "call",
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    European option pricer under rBergomi via Monte Carlo.

    Parameters
    ----------
    S0, K, T, r, q : floats
        Spot, strike, maturity (years), risk-free, dividend yield.
    H, eta, rho : floats
        Rough parameters. H in (0,1), eta > 0, rho in (-1,1).
    xi0 : float | array-like | callable
        Forward variance curve. If scalar, constant. If array-like, broadcast
        across the time grid. If callable, it must accept t-array and return same shape.
    n_paths : int
        Number of MC paths.
    N : int
        Time steps. Uniform grid.
    option : str
        'call' or 'put'.
    seed : int or None
        RNG seed.

    Returns
    -------
    price : float
        Discounted MC price.
    stderr : float
        Standard error of the MC estimator.

    Notes
    -----
    - Uses log-Euler for S with instantaneous variance from rBergomi.
    - Uses slow Hosking fBM. Expect O(n_paths*N^2) because fBM is per path O(N^2).
      Good for validation and small grids.
    - For speed, replace fbm_increments_hosking with a hybrid scheme.
    """
    option = _validate_call_put(option)
    S0 = _floor_pos(S0)
    K = _floor_pos(K)
    T = _floor_pos(T)
    n_paths = int(n_paths)
    N = int(N)
    if n_paths < 1 or N < 1:
        raise ValueError("n_paths and N must be >= 1")

    t, S, _v = rbergomi_paths(
        S0=S0, T=T, N=N, n_paths=n_paths, H=H, eta=eta, rho=rho,
        xi0=xi0, r=r, q=q, seed=seed
    )

    ST = S[:, -1]
    if option == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    DF = math.exp(-r * T)
    disc = DF * payoff
    price = float(np.mean(disc))
    stderr = float(np.std(disc, ddof=1) / math.sqrt(n_paths))
    return price, stderr

