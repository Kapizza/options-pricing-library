# svi_surface.py
# -----------------------------------------------------------------------------
# Raw-SVI per expiry + calendar no-arbitrage stitching
#   w(k) = a + b * [ rho*(k - m) + sqrt((k - m)^2 + sigma^2) ]
#
# Provides:
#   - fit_svi_expiry_from_ivs(...)
#   - fit_svi_expiry_from_prices(...)
#   - stitch_no_arb_calendar(...)
#   - fit_svi_surface(chains_by_expiry, S0, r, q, mode="auto")
#   - SVISurface (iv(k, T), w(k, T))
#
# Notes:
#   * Calendar no-arb: enforce monotonicity of w/T in T via isotonic regression,
#     evaluated on a fixed log-moneyness grid. This removes calendar arbitrage.
#   * Static butterfly no-arb: we apply feasibility constraints + a numeric
#     convexity check on w(k). It’s a pragmatic guard; for production you may
#     add the closed-form density-positivity conditions as well.
# -----------------------------------------------------------------------------

import math
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression

# Use the existing BS helpers (IV inversion) from black_scholes.py
from src.black_scholes import implied_vol_from_price  # robust Brent + Newton


# ------------------------------- SVI primitives -------------------------------

@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k_array, p):
    """
    Total variance w(k) under raw-SVI for an array-like of log-moneyness k.
    """
    k = np.asarray(k_array, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma * p.sigma))


def _feasible_raw_svi(p, eps=1e-12):
    """
    Lightweight feasibility checks often used in practice:
      - b >= 0
      - |rho| < 1
      - sigma > 0
      - lower bound on 'a' so that w(k) cannot dip below ~0 at the minimum
    """
    if p.b < 0 or p.sigma <= 0 or not (-0.999 < p.rho < 0.999):
        return False
    min_a = -p.b * p.sigma * math.sqrt(max(0.0, 1 - p.rho * p.rho)) - eps
    return p.a >= min_a


def _project_params(p):
    """
    Project parameters back to a feasible set (gentle nudges).
    """
    a = float(p.a)
    b = max(float(p.b), 1e-10)
    rho = max(min(float(p.rho), 0.999), -0.999)
    m = float(p.m)
    s = max(float(p.sigma), 1e-8)
    # lift 'a' if needed to guarantee nonnegativity bound
    min_a = -b * s * math.sqrt(max(0.0, 1 - rho * rho)) + 1e-12
    if a < min_a:
        a = min_a
    return SVIParams(a, b, rho, m, s)


def _numeric_convex(wk, k, tol=-1e-8):
    """
    Numeric convexity guard: finite-difference second derivative >= tol.
    A small negative tol allows tiny discretization noise.
    """
    k = np.asarray(k, dtype=float)
    wk = np.asarray(wk, dtype=float)
    if k.size < 5:
        return True  # not enough points to judge reliably
    h = np.gradient(k)
    wpp = (np.roll(wk, -1) - 2 * wk + np.roll(wk, 1)) / (
        (0.5 * (h + np.roll(h, 1))) ** 2 + 1e-16
    )
    wpp = wpp[1:-1]
    return np.all(wpp >= tol)


# -------------------------- Per-expiry SVI calibration ------------------------
def _convexify_row_in_k(k_grid, w_row):
    """
    Project a 1D curve w(k) onto the convex cone by
    isotonic regression on its discrete slopes, then integrate back.
    Works for non-uniform k spacing.
    """
    k = np.asarray(k_grid, float)
    w = np.asarray(w_row, float)
    dk = np.diff(k)
    s  = np.diff(w) / np.maximum(dk, 1e-12)  # discrete slopes

    # Enforce non-decreasing slopes via isotonic regression
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    # x positions don't matter for isotonic; use segment midpoints
    x_mid = 0.5 * (k[:-1] + k[1:])
    s_hat = ir.fit_transform(x_mid, s)

    # Integrate slopes back to a convex ŵ
    w_hat = np.empty_like(w)
    w_hat[0] = w[0]
    w_hat[1:] = w_hat[0] + np.cumsum(s_hat * dk)

    # Keep total variance non-negative
    return np.maximum(w_hat, 0.0)


def _huber(res, delta):
    """Huber loss (elementwise)."""
    a = np.abs(res)
    quad = 0.5 * (res ** 2)
    lin  = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)

def _map_unconstrained_to_svi(theta):
    # theta = (c, beta, rho_tilde, m, s)  -> (a,b,rho,m,sigma)
    c, beta, rho_tilde, m, s = [float(t) for t in theta]
    b = math.exp(beta)                 # >0
    rho = math.tanh(rho_tilde)         # (-1,1)
    sigma = math.exp(s)                # >0
    # ensure nonnegative total variance at the minimum
    a = (math.exp(c) ** 2) - b * sigma * math.sqrt(max(0.0, 1.0 - rho * rho)) + 1e-12
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

def _huber(res, delta):
    a = np.abs(res)
    quad = 0.5 * (res ** 2)
    lin  = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)

def fit_svi_expiry_from_ivs(K, iv, T, F):
    """
    Robust per-expiry SVI fit with feasibility mapping.
    Pipeline:
      1) Multi-start Nelder–Mead (Huber loss)
      2) L-BFGS-B polish (Huber)
      3) L-BFGS-B polish (pure MSE on total variance)
      4) 1-D Brent on 'c' (controls 'a') for final level alignment
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)
    iv = np.clip(iv, 1e-8, 5.0)

    T = float(max(T, 1e-8))
    F = float(max(F, 1e-12))
    k = np.log(K / F)
    w_tgt = (iv * iv) * T

    # --- stats & helpers ---
    k_med = float(np.median(k))
    k_lo, k_hi = np.percentile(k, [15, 85])
    w_lo, w_hi = np.percentile(w_tgt, [15, 85])
    w_med = float(np.median(w_tgt))

    def guess_b(sig0, m0):
        d1 = np.sqrt((k_hi - m0) ** 2 + sig0 * sig0)
        d0 = np.sqrt((k_lo - m0) ** 2 + sig0 * sig0)
        denom = (d1 - d0) if abs(d1 - d0) > 1e-10 else 1e-10
        return max(1e-4, (w_hi - w_lo) / denom)

    def loss_huber_theta(theta):
        p = _map_unconstrained_to_svi(theta)
        w = svi_total_variance(k, p)
        return float(np.mean(_huber(w - w_tgt, huber_delta)))

    def loss_mse_theta(theta):
        p = _map_unconstrained_to_svi(theta)
        w = svi_total_variance(k, p)
        r = w - w_tgt
        return float(np.mean(r * r))

    # --- starts: grid + jitters ---
    rng = np.random.RandomState(42)
    m_grid     = np.array([0.0, k_med], dtype=float)
    sigma_grid = np.array([0.10, 0.20, 0.40], dtype=float)
    rho_grid   = np.array([-0.6, -0.3, 0.0, 0.3, 0.6], dtype=float)

    starts = []
    for m0 in m_grid:
        for s0 in sigma_grid:
            b0 = guess_b(s0, m0)
            target_a = max(1e-6, 0.7 * w_med)
            # invert mapping approximately to pick c so that a ≈ target_a
            # a = exp(c)^2 - b*s*sqrt(1-rho^2) + eps  -> exp(c)^2 ≈ target_a + adj
            for rho0 in rho_grid:
                adj = b0 * s0 * math.sqrt(max(0.0, 1 - rho0 * rho0)) - 1e-12
                base = max(1e-12, target_a + adj)
                c0 = 0.5 * math.log(base)
                theta0 = np.array([c0, math.log(max(b0, 1e-4)), np.arctanh(rho0 * 0.999), m0, math.log(s0)], dtype=float)
                starts.append(theta0)

    # random jitters
    for _ in range(6):
        j = rng.normal(scale=[0.2, 0.3, 0.3, 0.15, 0.3], size=5)
        starts.append(starts[rng.randint(len(starts))] + j)

    # --- robust loss scale in w-units ---
    huber_delta = 2e-3  # slightly tighter than before

    # --- 1) global-ish NM on Huber ---
    best_theta, best_val = None, np.inf
    for theta0 in starts:
        res_nm = minimize(
            loss_huber_theta, theta0, method="Nelder-Mead",
            options=dict(maxiter=2000, xatol=1e-7, fatol=1e-10, adaptive=True)
        )
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_theta = res_nm.x

    # --- 2) L-BFGS-B polish on Huber ---
    res_lb_h = minimize(
        loss_huber_theta, best_theta, method="L-BFGS-B",
        options=dict(maxiter=500, ftol=1e-15)
    )
    theta = res_lb_h.x

    # --- 3) L-BFGS-B polish on MSE (tight alignment) ---
    res_lb_mse = minimize(
        loss_mse_theta, theta, method="L-BFGS-B",
        options=dict(maxiter=600, ftol=1e-15)
    )
    theta = res_lb_mse.x
    p = _map_unconstrained_to_svi(theta)

    # --- 4) 1-D Brent on 'c' only (level fine-tune) ---
    # keep (beta, rho_tilde, m, s) fixed, vary c to minimize pure MSE
    beta, rho_tilde, m_uncon, s_uncon = theta[1], theta[2], theta[3], theta[4]

    def loss_mse_c(c):
        p_loc = _map_unconstrained_to_svi([c, beta, rho_tilde, m_uncon, s_uncon])
        w = svi_total_variance(k, p_loc)
        r = w - w_tgt
        return float(np.mean(r * r))

    # bracket around current c
    c0 = float(theta[0])
    bracket = (c0 - 2.0, c0 + 2.0)
    from scipy.optimize import minimize_scalar
    res_c = minimize_scalar(loss_mse_c, bracket=bracket, method="brent")
    theta[0] = float(res_c.x)
    p = _map_unconstrained_to_svi(theta)

    # Convexity safeguard
    kk = np.linspace(k.min() - 0.5, k.max() + 0.5, 121)
    if not _numeric_convex(svi_total_variance(kk, p), kk):
        p = SVIParams(p.a, p.b, 0.92 * p.rho, p.m, 1.12 * p.sigma)

    return p

def fit_svi_expiry_from_prices(S, r, q, T, K, call_mid):
    """
    Fit raw-SVI from call mid prices at a single expiry.
    Internally inverts IVs using black_scholes.implied_vol_from_price.
    """
    S = float(S)
    r = float(r)
    q = float(q)
    T = float(T)
    K = np.asarray(K, dtype=float)
    call_mid = np.asarray(call_mid, dtype=float)

    # Convert prices -> IVs (call)
    ivs = np.empty_like(call_mid, dtype=float)
    for i in range(call_mid.size):
        ivs[i] = implied_vol_from_price(S, K[i], T, r - q, call_mid[i], option_type="call")
    F = S * math.exp((r - q) * T)
    return fit_svi_expiry_from_ivs(K, ivs, T, F)


# ----------------------- Calendar stitching (no-arbitrage) --------------------

@dataclass
class SVISurface:
    tenors: np.ndarray    # shape (M,)
    params: list          # list of SVIParams, len M
    k_grid: np.ndarray    # grid used for calendar projection (log-moneyness)
    w_grid: np.ndarray    # shape (M, K) total variance after projection (calendar no-arb)

    @lru_cache(maxsize=None)
    def _interp_T(self):
        # Prebuild T-interpolator over w_grid for speed.
        return interp1d(self.tenors, self.w_grid, axis=0, kind="linear", fill_value="extrapolate")

    def w(self, k_array, T):
        """
        Interpolate total variance to (k, T) using the precomputed no-arb grid.
        """
        k = np.asarray(k_array, dtype=float)
        # Interpolate across k within each tenor
        w_T = []
        for m in range(len(self.tenors)):
            f_k = interp1d(self.k_grid, self.w_grid[m], kind="linear", fill_value="extrapolate")
            w_T.append(f_k(k))
        w_T = np.vstack(w_T)  # (M, len(k))
        fT = interp1d(self.tenors, w_T, axis=0, kind="linear", fill_value="extrapolate")
        return fT(float(T))

    def iv(self, k_array, T):
        """
        Implied vol σ(k, T) from w(k, T) as sqrt(w / T).
        """
        T = float(max(T, 1e-8))
        w = self.w(k_array, T)
        return np.sqrt(np.maximum(w, 1e-12) / T)


def stitch_no_arb_calendar(tenors, fitted_params, k_grid):
    """
    Enforce calendar no-arbitrage by projecting total variance w(k, T)
    to be non-decreasing in T at each k on a fixed k_grid,
    then repair convexity in k (butterfly no-arb) per tenor.
    """
    tenors = np.asarray(tenors, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)

    M = tenors.size
    # raw w from per-expiry fits
    Wtot = np.zeros((M, k_grid.size), dtype=float)
    for i, (T, p) in enumerate(zip(tenors, fitted_params)):
        Wtot[i, :] = svi_total_variance(k_grid, p)

    # sort by T, project monotone-in-T per k
    idx = np.argsort(tenors)
    T_sorted = tenors[idx]
    W_sorted = Wtot[idx]

    irT = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for j in range(k_grid.size):
        W_sorted[:, j] = irT.fit_transform(T_sorted, W_sorted[:, j])

    # unsort
    W_mono = np.zeros_like(Wtot)
    W_mono[idx] = W_sorted

    # --- NEW: convexity repair in k per tenor ---
    for i in range(M):
        W_mono[i, :] = _convexify_row_in_k(k_grid, W_mono[i, :])

    return SVISurface(
        tenors=tenors.copy(),
        params=list(fitted_params),
        k_grid=k_grid.copy(),
        w_grid=W_mono,
    )



# ------------------------------- High-level entry -----------------------------

def fit_svi_surface(chains_by_expiry, S0, r, q, mode="auto"):
    """
    Fit an arbitrage-aware SVI surface from per-expiry option data.

    Parameters
    ----------
    chains_by_expiry : dict keyed by T (float years). Each value is a dict with:
        If mode in {"price","auto"}:  keys "K", "call_mid" (near-ATM calls are ok)
        If mode == "iv":              keys "K", "iv"
        You may also include "F" to skip F recompute (optional).
    S0, r, q : spot, risk-free rate, dividend yield (cont. comp.)
    mode     : "price" | "iv" | "auto"

    Returns
    -------
    SVISurface
    """
    # Collect and sort maturities
    tenors = sorted(chains_by_expiry.keys())
    fitted = []
    all_k = []

    for T in tenors:
        d = chains_by_expiry[T]
        K = np.asarray(d["K"], dtype=float)

        if mode == "iv" or ("iv" in d and mode == "auto"):
            iv = np.asarray(d["iv"], dtype=float)
            # Forward for this expiry
            F = d.get("F", float(S0 * math.exp((r - q) * T)))
            p = fit_svi_expiry_from_ivs(K, iv, T, F)
        else:
            call_mid = np.asarray(d["call_mid"], dtype=float)
            p = fit_svi_expiry_from_prices(S0, r, q, T, K, call_mid)

        fitted.append(p)

        # Keep track of pooled log-moneyness for grid design
        F_T = d.get("F", float(S0 * math.exp((r - q) * T)))
        k_slice = np.log(K / max(F_T, 1e-12))
        all_k.append(k_slice)

    all_k = np.concatenate(all_k)
    lo, hi = np.percentile(all_k, [1, 99])
    k_grid = np.linspace(lo - 0.5, hi + 0.5, 121)

    return stitch_no_arb_calendar(np.asarray(tenors, dtype=float), fitted, k_grid)
