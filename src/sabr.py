# sabr.py
# Lognormal SABR (Hagan 2002) with BS mapping.

import numpy as np
from scipy.optimize import minimize

from src.black_scholes import (
    black_scholes_price,
    bs_call_forward, bs_put_forward,
    implied_vol_from_price,  
)

__all__ = [
    "sabr_iv",
    "sabr_price",
    "sabr_calibrate_iv",
    "sabr_calibrate_price",
]

# -----------------------------
# Hagan (2002) lognormal SABR IV
# -----------------------------

def sabr_iv(F, K, T, alpha, beta, rho, nu, eps=1e-12):
    """
    Hagan et al. (2002) asymptotic implied volatility (lognormal SABR).
    Returns Black–Scholes IV given forward F and strike K.
    """
    F = float(F); K = float(K); T = float(T)
    alpha = float(alpha); beta = float(beta); rho = float(rho); nu = float(nu)

    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0 or nu < 0:
        return np.nan

    one_m_beta = 1.0 - beta
    FK_beta = (F * K)**(0.5 * one_m_beta) if one_m_beta != 0 else 1.0
    logFK = np.log(F / K) if K > 0 else 0.0

    if abs(F - K) < eps:
        z = 0.0
    else:
        z = (nu / alpha) * FK_beta * logFK

    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    num = sqrt_term + z - rho
    den = 1.0 - rho
    if num <= 0 or den <= 0:
        x = np.log(max(num, eps) / max(den, eps))
    else:
        x = np.log(num / den)

    z_over_x = 1.0 if abs(z) < eps else z / x

    # Hagan prefactor and correction
    if one_m_beta != 0:
        F_pow = F**(one_m_beta)
        K_pow = K**(one_m_beta)
        denom = np.sqrt(F_pow * K_pow)
        A0 = alpha / max(denom, eps)
        term1 = (one_m_beta**2 / 24.0) * (alpha * alpha) / max(F_pow * K_pow, eps)
        term2 = (rho * beta * nu * alpha) / (4.0 * max(denom, eps))
    else:
        # beta == 1 limit
        A0 = alpha
        term1 = 0.0
        term2 = 0.0
    term3 = ((2.0 - 3.0 * rho * rho) * nu * nu) / 24.0

    A = A0 * (1.0 + (term1 + term2 + term3) * T)

    if abs(F - K) < eps:
        return float(A)          # ATM limit
    else:
        return float(A * z_over_x)

# -----------------------------
# SABR -> BS pricing
# -----------------------------

def sabr_price(S, K, T, r, q, alpha, beta, rho, nu, option="call"):
    """
    Price via SABR->BS: F = S*exp((r-q)T), DF = exp(-rT), BS with iv = sabr_iv(F,K,...).
    """
    if T <= 0:
        intrinsic = S - K if option == "call" else K - S
        return float(max(intrinsic, 0.0))
    DF = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    iv = sabr_iv(F, K, T, alpha, beta, rho, nu)
    undisc = bs_call_forward(F, K, T, iv) if option == "call" else bs_put_forward(F, K, T, iv)
    return float(DF * undisc)

# -----------------------------
# Calibration (IV-space)
# -----------------------------

def _bounds_ok(alpha, beta, rho, nu):
    return (alpha > 0) and (0.0 <= beta <= 1.0) and (-0.999 < rho < 0.999) and (nu >= 0)

def sabr_calibrate_iv(K, T, iv_mkt, F, w=None, beta=None, x0=None, bounds=None, maxiter=400, disp=False):
    """
    Calibrate SABR to market IVs at fixed forward(s) F (scalar or array).
    """
    K = np.asarray(K, float)
    T = np.asarray(T, float)
    iv_mkt = np.asarray(iv_mkt, float)
    if np.ndim(F) == 0:
        F = np.full_like(K, float(F))
    else:
        F = np.asarray(F, float)
    w = np.ones_like(K, float) if w is None else np.asarray(w, float)

    m = (K > 0) & (T > 0) & (F > 0) & np.isfinite(iv_mkt)
    K, T, F, iv_mkt, w = K[m], T[m], F[m], iv_mkt[m], w[m]

    if beta is None:
        if x0 is None:
            i0 = np.argmin(np.abs(K - F)) if len(K) else 0
            atm_iv = float(iv_mkt[i0]) if len(iv_mkt) else 0.2
            x0 = np.array([max(1e-3, atm_iv), 0.7, -0.2, 0.5])  # alpha, beta, rho, nu
        if bounds is None:
            bounds = [(1e-6, 5.0), (0.0, 1.0), (-0.999, 0.999), (1e-6, 5.0)]

        def obj(x):
            a, b, r_, n_ = x
            if not _bounds_ok(a, b, r_, n_): return 1e12
            iv_model = np.array([sabr_iv(Fi, Ki, Ti, a, b, r_, n_) for Fi, Ki, Ti in zip(F, K, T)])
            d = iv_model - iv_mkt
            return float(np.mean(w * d * d))

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter, disp=disp))
        return dict(alpha=res.x[0], beta=res.x[1], rho=res.x[2], nu=res.x[3]), res
    else:
        bfix = float(beta)
        if x0 is None:
            i0 = np.argmin(np.abs(K - F)) if len(K) else 0
            atm_iv = float(iv_mkt[i0]) if len(iv_mkt) else 0.2
            x0 = np.array([max(1e-3, atm_iv), -0.2, 0.5])  # alpha, rho, nu
        if bounds is None:
            bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]

        def obj(x):
            a, r_, n_ = x
            if not _bounds_ok(a, bfix, r_, n_): return 1e12
            iv_model = np.array([sabr_iv(Fi, Ki, Ti, a, bfix, r_, n_) for Fi, Ki, Ti in zip(F, K, T)])
            d = iv_model - iv_mkt
            return float(np.mean(w * d * d))

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter, disp=disp))
        return dict(alpha=res.x[0], beta=bfix, rho=res.x[1], nu=res.x[2]), res

# -----------------------------
# Calibration (price-space)
# -----------------------------

def sabr_calibrate_price(S, r, q, K, T, price_mkt, side, w=None, beta=None, x0=None, bounds=None, maxiter=400, disp=False):
    """
    Calibrate SABR by minimizing price errors (maps SABR -> BS via sabr_iv).
    """
    S = float(S) if np.ndim(S) == 0 else np.asarray(S, float)
    r = float(r) if np.ndim(r) == 0 else np.asarray(r, float)
    q = float(q) if np.ndim(q) == 0 else np.asarray(q, float)
    K = np.asarray(K, float); T = np.asarray(T, float)
    price_mkt = np.asarray(price_mkt, float); side = np.asarray(side)
    n = len(K)
    if np.ndim(S) == 0: S = np.full(n, S)
    if np.ndim(r) == 0: r = np.full(n, r)
    if np.ndim(q) == 0: q = np.full(n, q)
    w = np.ones(n, float) if w is None else np.asarray(w, float)

    m = (K > 0) & (T > 0) & (S > 0) & np.isfinite(price_mkt)
    S, r, q, K, T, price_mkt, side, w = S[m], r[m], q[m], K[m], T[m], price_mkt[m], side[m], w[m]

    if beta is None:
        if x0 is None:
            x0 = np.array([0.2, 0.7, -0.2, 0.5])  # alpha, beta, rho, nu
        if bounds is None:
            bounds = [(1e-6, 5.0), (0.0, 1.0), (-0.999, 0.999), (1e-6, 5.0)]

        def obj(x):
            a, b, r_, n_ = x
            if not _bounds_ok(a, b, r_, n_): return 1e12
            loss, wsum = 0.0, 0.0
            for Si, ri, qi, Ki, Ti, Pi, sd, wi in zip(S, r, q, K, T, price_mkt, side, w):
                DF = np.exp(-ri * Ti); Fi = Si * np.exp((ri - qi) * Ti)
                iv = sabr_iv(Fi, Ki, Ti, a, b, r_, n_)
                model = DF * (bs_call_forward(Fi, Ki, Ti, iv) if sd == "call" else bs_put_forward(Fi, Ki, Ti, iv))
                loss += wi * (model - Pi)**2; wsum += wi
            return loss / max(wsum, 1e-12)

        opts = dict(maxiter=maxiter)
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options=opts)

        return dict(alpha=res.x[0], beta=res.x[1], rho=res.x[2], nu=res.x[3]), res
    else:
        bfix = float(beta)
        if x0 is None:
            x0 = np.array([0.2, -0.2, 0.5])  # alpha, rho, nu
        if bounds is None:
            bounds = [(1e-6, 5.0), (-0.999, 0.999), (1e-6, 5.0)]

        def obj(x):
            a, r_, n_ = x
            if not _bounds_ok(a, bfix, r_, n_): return 1e12
            loss, wsum = 0.0, 0.0
            for Si, ri, qi, Ki, Ti, Pi, sd, wi in zip(S, r, q, K, T, price_mkt, side, w):
                DF = np.exp(-ri * Ti); Fi = Si * np.exp((ri - qi) * Ti)
                iv = sabr_iv(Fi, Ki, Ti, a, bfix, r_, n_)
                model = DF * (bs_call_forward(Fi, Ki, Ti, iv) if sd == "call" else bs_put_forward(Fi, Ki, Ti, iv))
                loss += wi * (model - Pi)**2; wsum += wi
            return loss / max(wsum, 1e-12)

        opts = dict(maxiter=maxiter)
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options=opts)

        return dict(alpha=res.x[0], beta=bfix, rho=res.x[1], nu=res.x[2]), res
