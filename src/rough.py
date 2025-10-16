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
import os
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



# ---------- parallel utilities ----------

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from numpy.random import SeedSequence

def _split_batches(n, batch_size):
    n = int(n); batch_size = int(max(1, batch_size))
    sizes = []
    done = 0
    while done < n:
        take = min(batch_size, n - done)
        sizes.append(take)
        done += take
    return sizes

def _child_seeds(base_seed, n_children):
    ss = SeedSequence(int(base_seed))
    kids = ss.spawn(int(n_children))
    # return raw ints for np.random.default_rng
    return [int(k.generate_state(1)[0]) for k in kids]


# Try to avoid thread oversubscription when we parallelize at Python level.
# Respect existing env if the user already configured them.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = "1"

# Optional Numba acceleration
try:  # pragma: no cover - optional dependency
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None
    prange = range  # fallback

_HAS_NUMBA = njit is not None




#------------------------ Davies–Harte fBm ------------------------
def fbm_davies_harte(N, H, n_paths, rng):
    N = int(N)
    n_paths = int(n_paths)
    if N < 1 or n_paths < 1:
        raise ValueError("N and n_paths must be >= 1")
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0,1)")

    m = np.arange(0, N, dtype=float)
    r = 0.5 * (np.power(m + 1.0, 2.0 * H) - 2.0 * np.power(m, 2.0 * H) + np.power(np.abs(m - 1.0), 2.0 * H))
    r[0] = 1.0

    c = np.empty(2 * N, dtype=float)
    c[:N] = r
    c[N] = 0.0
    c[N+1:] = r[1:][::-1]

    lam = np.fft.fft(c).real
    lam = np.maximum(lam, 0.0)
    sqrtlam = np.sqrt(lam)

    Z = np.empty((n_paths, 2 * N), dtype=np.complex128)
    Z[:, 0] = rng.normal(0.0, 1.0, size=n_paths)
    Z[:, N] = rng.normal(0.0, 1.0, size=n_paths)
    for k in range(1, N):
        a = rng.normal(0.0, 1.0, size=n_paths)
        b = rng.normal(0.0, 1.0, size=n_paths)
        Z[:, k] = a + 1j * b
        Z[:, 2 * N - k] = np.conj(Z[:, k])

    Y = Z * sqrtlam[None, :]
    fgn = np.fft.ifft(Y, axis=1).real * np.sqrt(2 * N)   # unit-step Var for fGn
    fgn = fgn[:, :N]

    B = np.cumsum(fgn, axis=1)
    B = np.hstack([np.zeros((n_paths, 1), dtype=float), B])  # (paths, N+1)

    return B

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



# ------------------------ fast fGn via Davies–Harte (FFT) ------------------------
if 0:

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _dh_rfft_eigs(N, H):
        # first row of the 2N circulant covariance for fGn
        def gamma(k):
            k = abs(int(k))
            if k == 0:
                return 1.0
            return 0.5 * ((k + 1)**(2*H) - 2*(k**(2*H)) + (k - 1)**(2*H))

        N = int(N)
        if N < 1:
            raise ValueError("N must be >= 1")
        if not (0.0 < H < 1.0):
            raise ValueError("H must be in (0, 1)")

        g = np.array([gamma(k) for k in range(N)], dtype=float)
        c = np.empty(2*N, dtype=float)
        c[:N] = g
        c[N] = 0.0
        c[N+1:] = g[1:][::-1]

        lam = np.fft.rfft(c).real   # length N+1, nonnegative up to rounding
        lam = np.maximum(lam, 0.0)
        return lam  # rfft eigenvalues


    def fgn_davies_harte(N, H, n_paths, rng):
        N = int(N)
        M = 2 * N
        lam = _dh_rfft_eigs(N, H)            # length N+1 rfft eigenvalues

        U = rng.standard_normal((n_paths, N+1))
        V = rng.standard_normal((n_paths, N+1))

        Y = np.empty((n_paths, N+1), dtype=np.complex128)
        Y[:, 0] = np.sqrt(lam[0]) * U[:, 0]
        Y[:, N] = np.sqrt(lam[N]) * U[:, N] if N > 0 else 0.0

        if N > 1:
            scale = np.sqrt(lam[1:N] / 2.0)[None, :]
            Y[:, 1:N] = (U[:, 1:N] + 1j * V[:, 1:N]) * scale

        # NumPy irfft has a 1/M factor; compensate by sqrt(M)
        x_full = np.fft.irfft(Y, n=M, axis=1).real * np.sqrt(M)
        x = x_full[:, :N]
        s = float(np.std(x[:, 0], ddof=1))
        if s > 0:
            x /= s
        return x


# ------------------------ rBergomi paths ------------------------

def rbergomi_paths(
    S0, T, N, n_paths, H, eta, rho, xi0,
    r=0.0, q=0.0, seed=None, fgn_method="davies-harte"
):
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
    S0 = _floor_pos(S0); T = _floor_pos(T)
    N = int(N); n_paths = int(n_paths)
    if N < 1 or n_paths < 1: raise ValueError("N and n_paths must be >= 1")
    if not (-0.999 < rho < 0.999): raise ValueError("rho must be in (-0.999, 0.999)")
    if not (0.0 < H < 1.0): raise ValueError("H must be in (0,1)")
    eta = _floor_pos(eta); r = float(r); q = float(q)

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, N + 1)
    dt = T / N
    sqrt_dt = math.sqrt(dt)

    xi_fn = _xi0_as_callable(xi0)
    xi_vec = xi_fn(t)

    # Build W^H(t)
    if fgn_method.lower().startswith("dav"):
        BH = fbm_davies_harte(N, H, n_paths, rng)      # unit-step fBm
        W_H = BH * (dt**H)
    else:
        W_H = np.empty((n_paths, N + 1), dtype=float); W_H[:, 0] = 0.0
        for i in range(n_paths):
            fgn = fbm_increments_hosking(N, H, rng)
            W_H[i, 1:] = np.cumsum(fgn) * (dt**H)

    # One global normalization (self-similarity fixes all t)
    var_emp  = float(np.var(W_H[:, -1], ddof=1))
    var_theo = float(t[-1]**(2.0 * H))
    if var_emp > 0.0 and var_emp != var_theo:
        W_H *= math.sqrt(var_theo / var_emp)

    # Correlated Brownian for S
    Z1 = rng.standard_normal((n_paths, N))
    Z2 = rng.standard_normal((n_paths, N))
    dW_S = rho * Z2 + math.sqrt(1.0 - rho*rho) * Z1

    t_pow = t**(2.0 * H)
    drift_corr = -0.5 * (eta**2) * t_pow
    v = (xi_vec[None, :] * np.exp(eta * W_H + drift_corr[None, :])).astype(float)
    v = np.maximum(v, 1e-14)

    S = np.empty((n_paths, N + 1), dtype=float)
    S[:, 0] = S0
    drift = (r - q) * dt
    for k in range(N):
        vol_step = np.sqrt(np.maximum(v[:, k], 1e-14)) * sqrt_dt
        S[:, k + 1] = S[:, k] * np.exp(drift - 0.5 * vol_step**2 + vol_step * dW_S[:, k])

    return t, S, v



# ---------- rBergomi parallel wrapper ----------

def _rbergomi_worker(args):
    # separate function so it's pickleable by ProcessPool on Windows
    (S0,T,N,n_i,H,eta,rho,xi0,r,q,seed,fgn_method) = args
    return rbergomi_paths(
        S0=S0, T=T, N=N, n_paths=n_i, H=H, eta=eta, rho=rho,
        xi0=xi0, r=r, q=q, seed=seed, fgn_method=fgn_method
    )

def _rbergomi_terminal_worker(args):
    # Returns only terminal values to reduce IPC costs
    (S0,T,N,n_i,H,eta,rho,xi0,r,q,seed,fgn_method) = args
    t, S, _v = rbergomi_paths(
        S0=S0, T=T, N=N, n_paths=n_i, H=H, eta=eta, rho=rho,
        xi0=xi0, r=r, q=q, seed=seed, fgn_method=fgn_method
    )
    return S[:, -1]

def rbergomi_paths_parallel(
    S0, T, N, n_paths, H, eta, rho, xi0,
    r=0.0, q=0.0, base_seed=12345, fgn_method="davies-harte",
    n_workers=4, batch_size=8192, return_variance=True
):
    """
    Parallel rBergomi: split n_paths into batches and run in processes.

    Notes:
    - Set MKL_NUM_THREADS=1 and OMP_NUM_THREADS=1 in your shell to avoid overthreading.
    - On Windows, wrap calls in 'if __name__ == "__main__":' when running as a script.
    """
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))

    tasks = []
    for n_i, s in zip(sizes, seeds):
        tasks.append((S0, T, N, n_i, H, eta, rho, xi0, r, q, s, fgn_method))

    with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
        outs = list(ex.map(_rbergomi_worker, tasks))

    # merge
    t = outs[0][0]
    S = np.vstack([o[1] for o in outs])
    if return_variance:
        v = np.vstack([o[2] for o in outs])
        return t, S, v
    return t, S

def rbergomi_paths_parallel_pool(
    executor,
    S0, T, N, n_paths, H, eta, rho, xi0,
    r=0.0, q=0.0, base_seed=12345, fgn_method="davies-harte",
    batch_size=8192, return_variance=True
):
    """Same as rbergomi_paths_parallel but reuses a provided executor."""
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    tasks = [(S0, T, N, n_i, H, eta, rho, xi0, r, q, s, fgn_method) for n_i, s in zip(sizes, seeds)]
    outs = list(executor.map(_rbergomi_worker, tasks))
    t = outs[0][0]
    S = np.vstack([o[1] for o in outs])
    if return_variance:
        v = np.vstack([o[2] for o in outs])
        return t, S, v
    return t, S

def rbergomi_terminal_parallel_pool(
    executor,
    S0, T, N, n_paths, H, eta, rho, xi0,
    r=0.0, q=0.0, base_seed=12345, fgn_method="davies-harte",
    batch_size=8192
):
    """Parallel rBergomi returning only terminal ST to minimize IPC."""
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    tasks = [(S0, T, N, n_i, H, eta, rho, xi0, r, q, s, fgn_method) for n_i, s in zip(sizes, seeds)]
    outs = list(executor.map(_rbergomi_terminal_worker, tasks))
    ST = np.concatenate(outs, axis=0)
    return ST


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



def _validate_option(opt):
    s=str(opt).lower()
    if s not in ("call","put"): raise ValueError("option must be 'call' or 'put'")
    return s
def _gamma(x): return math.gamma(float(x))

def _kernel_weights(H, N, dt):
    H=float(H); N=int(N)
    if not (0.0 < H < 1.0): raise ValueError("H must be in (0,1)")
    if N<1: raise ValueError("N must be >= 1")
    m = np.arange(0, N+1, dtype=float)
    alpha = np.zeros(N+1, dtype=float)
    if N>=1: alpha[1:] = m[1:]**(H+0.5) - m[:-1]**(H+0.5)
    cH = 1.0/_gamma(H+0.5)
    drift_scale = cH * (dt**(H+0.5))
    diff_scale  = cH * (dt**H)
    return cH, alpha, drift_scale, diff_scale

def _rough_heston_paths_python(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho, r=0.0, q=0.0, seed=None, batch_size=1024):
    S0=_floor_pos(S0); v0=max(float(v0),0.0); T=_floor_pos(T)
    N=int(N); n_paths=int(n_paths)
    if N<1 or n_paths<1: raise ValueError("N and n_paths must be >= 1")
    if not (-0.999 < rho < 0.999): raise ValueError("rho must be in (-0.999,0.999)")
    if not (0.0 < float(H) < 1.0): raise ValueError("H must be in (0,1)")
    kappa=float(kappa); theta=max(float(theta),0.0); eta=_floor_pos(eta)
    r=float(r); q=float(q); batch_size=int(max(1,batch_size))

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, N+1); dt = T/N; sqrt_dt = math.sqrt(dt)
    _, alpha, drift_scale, diff_scale = _kernel_weights(H, N, dt)

    S = np.empty((n_paths, N+1)); V = np.empty((n_paths, N+1))
    S[:,0]=S0; V[:,0]=v0

    for start in range(0, n_paths, batch_size):
        end=min(n_paths,start+batch_size); m=end-start
        Z1 = rng.standard_normal((m,N))
        Z2 = rng.standard_normal((m,N))
        dW_S = rho*Z2 + math.sqrt(1.0-rho*rho)*Z1

        Sb = np.empty((m,N+1)); Vb = np.empty((m,N+1))
        Sb[:,0]=S0; Vb[:,0]=v0

        for k in range(N):
            Vk = np.maximum(Vb[:,k],1e-14); sqrtVk = np.sqrt(Vk)

            idx = np.arange(k+1); mlag=(k+1)-idx
            a = alpha[mlag]
            V_hist  = Vb[:,:k+1]
            Z2_hist = Z2[:,:k+1]
            sqrtV   = np.sqrt(np.maximum(V_hist,1e-14))
            drift_conv = np.dot(kappa*(theta - V_hist), a)
            diff_conv  = np.dot(sqrtV * Z2_hist, a)

            V_next = v0 + drift_scale*drift_conv + eta*diff_scale*diff_conv
            V_next = np.maximum(V_next,1e-14)
            Vb[:,k+1]=V_next

            Sb[:,k+1] = Sb[:,k] * np.exp((r-q)*dt - 0.5*Vk*dt + sqrtVk*sqrt_dt*dW_S[:,k])

        S[start:end,:]=Sb; V[start:end,:]=Vb

    return t, S, V


if _HAS_NUMBA:  # pragma: no cover - optional dependency

    @njit(parallel=True)
    def _rough_heston_kernel(S0, v0, dt, sqrt_dt, N, rho, r, q, kappa, theta, eta,
                             alpha, drift_scale, diff_scale, Z1, Z2, S_out, V_out):
        n_paths = S_out.shape[0]
        sqrt1mrho2 = math.sqrt(max(1.0 - rho * rho, 0.0))
        for i in prange(n_paths):
            S_out[i, 0] = S0
            V_out[i, 0] = v0
            for k in range(N):
                Vk_raw = V_out[i, k]
                Vk = Vk_raw if Vk_raw > 1e-14 else 1e-14
                sqrtVk = math.sqrt(Vk)

                drift_conv = 0.0
                diff_conv = 0.0
                for j in range(k + 1):
                    a = alpha[k + 1 - j]
                    v_hist = V_out[i, j]
                    drift_conv += kappa * (theta - v_hist) * a
                    v_hist_clamped = v_hist if v_hist > 1e-14 else 1e-14
                    diff_conv += math.sqrt(v_hist_clamped) * Z2[i, j] * a

                V_next = v0 + drift_scale * drift_conv + eta * diff_scale * diff_conv
                if V_next < 1e-14:
                    V_next = 1e-14
                V_out[i, k + 1] = V_next

                dW_S = rho * Z2[i, k] + sqrt1mrho2 * Z1[i, k]
                S_out[i, k + 1] = S_out[i, k] * math.exp((r - q) * dt - 0.5 * Vk * dt + sqrtVk * sqrt_dt * dW_S)

    def _rough_heston_paths_numba(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                                  r=0.0, q=0.0, seed=None, batch_size=1024):
        S0 = _floor_pos(S0); v0 = max(float(v0), 0.0); T = _floor_pos(T)
        N = int(N); n_paths = int(n_paths)
        if N < 1 or n_paths < 1:
            raise ValueError("N and n_paths must be >= 1")
        if not (-0.999 < rho < 0.999):
            raise ValueError("rho must be in (-0.999,0.999)")
        if not (0.0 < float(H) < 1.0):
            raise ValueError("H must be in (0,1)")
        kappa = float(kappa); theta = max(float(theta), 0.0); eta = _floor_pos(eta)
        r = float(r); q = float(q); batch_size = int(max(1, batch_size))

        rng = np.random.default_rng(seed)
        t = np.linspace(0.0, T, N + 1)
        dt = T / N
        sqrt_dt = math.sqrt(dt)
        _, alpha, drift_scale, diff_scale = _kernel_weights(H, N, dt)
        alpha = np.ascontiguousarray(alpha, dtype=float)

        S = np.empty((n_paths, N + 1), dtype=float)
        V = np.empty((n_paths, N + 1), dtype=float)

        for start in range(0, n_paths, batch_size):
            end = min(n_paths, start + batch_size)
            m = end - start
            Z1 = rng.standard_normal((m, N))
            Z2 = rng.standard_normal((m, N))
            Sb = np.empty((m, N + 1), dtype=float)
            Vb = np.empty((m, N + 1), dtype=float)
            _rough_heston_kernel(S0, v0, dt, sqrt_dt, N, float(rho), float(r), float(q),
                                 float(kappa), float(theta), float(eta),
                                 alpha, float(drift_scale), float(diff_scale),
                                 Z1, Z2, Sb, Vb)
            S[start:end, :] = Sb
            V[start:end, :] = Vb

        return t, S, V

else:

    def _rough_heston_paths_numba(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Numba is not available")


def _resolve_use_numba(flag):
    if flag is None:
        return _HAS_NUMBA
    return bool(flag) and _HAS_NUMBA


def rough_heston_paths(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                       r=0.0, q=0.0, seed=None, batch_size=1024, use_numba=None):
    """
    Rough Heston path generator with optional Numba acceleration.
    """
    use_numba_flag = _resolve_use_numba(use_numba)
    if use_numba_flag:
        return _rough_heston_paths_numba(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                                         r=r, q=q, seed=seed, batch_size=batch_size)
    return _rough_heston_paths_python(S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
                                      r=r, q=q, seed=seed, batch_size=batch_size)


# ---------- Rough Heston parallel wrapper ----------

def _rough_heston_worker(args):
    (S0,v0,T,N,n_i,H,kappa,theta,eta,rho,r,q,seed,batch_size,use_numba) = args
    return rough_heston_paths(
        S0=S0, v0=v0, T=T, N=N, n_paths=n_i, H=H, kappa=kappa, theta=theta,
        eta=eta, rho=rho, r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba
    )

def _rough_heston_terminal_worker(args):
    (S0,v0,T,N,n_i,H,kappa,theta,eta,rho,r,q,seed,batch_size,use_numba) = args
    t, S, V = rough_heston_paths(
        S0=S0, v0=v0, T=T, N=N, n_paths=n_i, H=H, kappa=kappa, theta=theta,
        eta=eta, rho=rho, r=r, q=q, seed=seed, batch_size=batch_size, use_numba=use_numba
    )
    return S[:, -1]

def rough_heston_paths_parallel(
    S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
    r=0.0, q=0.0, base_seed=12345, n_workers=4, batch_size=4096, use_numba=None
):
    """
    Parallel rough-Heston: split n_paths into process batches.
    """
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    use_numba_flag = _resolve_use_numba(use_numba)

    tasks = []
    for n_i, s in zip(sizes, seeds):
        tasks.append((S0, v0, T, N, n_i, H, kappa, theta, eta, rho, r, q, s, batch_size, use_numba_flag))

    with ProcessPoolExecutor(max_workers=int(n_workers)) as ex:
        outs = list(ex.map(_rough_heston_worker, tasks))

    # merge
    t = outs[0][0]
    S = np.vstack([o[1] for o in outs])
    V = np.vstack([o[2] for o in outs])
    return t, S, V

def rough_heston_paths_parallel_pool(
    executor,
    S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
    r=0.0, q=0.0, base_seed=12345, batch_size=4096, use_numba=None
):
    """Same as rough_heston_paths_parallel but reuses a provided executor."""
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    use_numba_flag = _resolve_use_numba(use_numba)
    tasks = [(S0, v0, T, N, n_i, H, kappa, theta, eta, rho, r, q, s, batch_size, use_numba_flag) for n_i, s in zip(sizes, seeds)]
    outs = list(executor.map(_rough_heston_worker, tasks))
    t = outs[0][0]
    S = np.vstack([o[1] for o in outs])
    V = np.vstack([o[2] for o in outs])
    return t, S, V

def rough_heston_terminal_parallel_pool(
    executor,
    S0, v0, T, N, n_paths, H, kappa, theta, eta, rho,
    r=0.0, q=0.0, base_seed=12345, batch_size=4096, use_numba=None
):
    """Parallel rough-Heston returning only terminal ST to minimize IPC."""
    sizes = _split_batches(n_paths, batch_size)
    seeds = _child_seeds(base_seed, len(sizes))
    use_numba_flag = _resolve_use_numba(use_numba)
    tasks = [(S0, v0, T, N, n_i, H, kappa, theta, eta, rho, r, q, s, batch_size, use_numba_flag) for n_i, s in zip(sizes, seeds)]
    outs = list(executor.map(_rough_heston_terminal_worker, tasks))
    ST = np.concatenate(outs, axis=0)
    return ST


def rough_heston_euro_mc(S0, v0, K, T, N, n_paths, H, kappa, theta, eta, rho, r=0.0, q=0.0, option="call", seed=None, batch_size=1024):
    option=_validate_option(option); K=_floor_pos(K)
    if N < 1 or n_paths < 1:
        raise ValueError("N and n_paths must be >= 1")
    if K <= 0.0:
        raise ValueError("K must be positive")
    t,S,V = rough_heston_paths(S0,v0,T,N,n_paths,H,kappa,theta,eta,rho,r,q,seed,batch_size)
    ST = S[:,-1]
    payoff = np.maximum(ST-K,0.0) if option=="call" else np.maximum(K-ST,0.0)
    DF = math.exp(-r*T); disc = DF*payoff
    price = float(np.mean(disc)); stderr = float(np.std(disc, ddof=1)/math.sqrt(S.shape[0]))
    return price, stderr

