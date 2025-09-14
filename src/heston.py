# options_pricing/heston.py
# Heston European option pricing via strike-centered COS (Fang & Oosterlee, 2008)
# with a safe Black–Scholes fallback in the near-constant-variance regime.

import numpy as np
from .black_scholes import black_scholes_price  # fallback in sigma→0 regime

__all__ = ["heston_charfunc", "heston_price", "heston_call_put"]

def heston_charfunc(u, T, r, kappa, theta, sigma, v0, rho, S0=1.0):
    """
    Risk-neutral characteristic function of X_T = ln S_T.
    Returns E[exp(i u X_T)].
    """
    i = 1j
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * i * u - b)**2 + (sigma**2) * (i*u + u**2))
    g = (b - rho*sigma*i*u - d) / (b - rho*sigma*i*u + d)
    exp_negdT = np.exp(-d*T)
    C = i*u*(np.log(S0) + r*T) + (a/(sigma**2)) * (
        (b - rho*sigma*i*u - d)*T - 2.0*np.log((1 - g*exp_negdT)/(1 - g))
    )
    D = ((b - rho*sigma*i*u - d)/(sigma**2)) * ((1 - exp_negdT)/(1 - g*exp_negdT))
    return np.exp(C + D*v0)

def _cumulants_x(T, r, kappa, theta, sigma, v0, rho, S0):
    """
    First two cumulants of X = ln S_T (approx; sufficient for COS truncation).
    """
    c1 = (np.log(S0) + r*T
          - 0.5*theta*T
          + (theta - v0)*(1 - np.exp(-kappa*T))/(2*kappa)
          - (rho*sigma*theta/(2*kappa))*(1 - np.exp(-kappa*T)))
    term1 = (sigma**2) * (1 - np.exp(-kappa*T)) * (
        v0*(kappa*T - 1 + np.exp(-kappa*T))/(kappa**2)
        + theta*(T - (2*(1 - np.exp(-kappa*T)))/kappa
                 + (1 - np.exp(-2*kappa*T))/(2*kappa))
    )
    c2 = max(1e-12, term1) / (kappa**2)
    return c1, c2

def _cos_coefficients_call_y(a, b, k):
    """
    COS coefficients for payoff G(y) = (e^y - 1)^+ on y ∈ [a, b].
    Handles k = 0 safely. Exercise region is y ∈ [max(0,a), b].
    """
    k = np.asarray(k, dtype=float)
    omega = k * np.pi / (b - a)

    c = max(0.0, a)
    d = b
    if d <= c:
        return np.zeros_like(omega)

    def psi(d_, c_):
        num = np.sin(omega*(d_ - a)) - np.sin(omega*(c_ - a))
        out = np.empty_like(omega, dtype=float)
        nz = omega != 0
        out[nz] = num[nz] / omega[nz]
        out[~nz] = (d_ - c_)  # ω → 0
        return out

    def chi(d_, c_):
        num = (np.cos(omega*(d_ - a)) * np.exp(d_) - np.cos(omega*(c_ - a)) * np.exp(c_)
               + omega * (np.sin(omega*(d_ - a)) * np.exp(d_) - np.sin(omega*(c_ - a)) * np.exp(c_)))
        den = (1.0 + omega**2)
        return num / den

    Vk = (2.0 / (b - a)) * (chi(d, c) - psi(d, c))
    return Vk

def _bs_fallback_if_constant_variance(kappa, theta, sigma, v0, rho):
    """
    Heuristic detector for the near-constant-variance regime where Heston ≈ BS.
    Tuned conservatively to avoid false positives.
    """
    if sigma < 5e-4 and abs(v0 - theta) < 1e-8 and abs(rho) < 1e-6 and kappa >= 5.0:
        return True
    return False

def heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho,
                 option="call", N=4096, L=12, alpha=None, umax=None, **kwargs):
    """
    European option price under Heston via strike-centered COS.
    y := ln(S_T / K). Price(call) = e^{-rT} * K * sum Re[phi_y(u_k) * Vk]
    where phi_y(u) = e^{-i u ln K} * phi_x(u), u_k = k*pi/(b-a).

    Accepts alpha, umax for API compatibility (unused).
    """
    if T <= 0:
        payoff = max(S0 - K, 0.0) if option == "call" else max(K - S0, 0.0)
        return float(payoff)
    if K <= 0.0:
        raise ValueError("Strike must be positive.")

    # --- Compute cumulants for ln S_T (used both for fallback & COS window) ---
    c1_x, c2_x = _cumulants_x(T, r, kappa, theta, sigma, v0, rho, S0)
    std2 = float(abs(c2_x))

    # --- Fallback: near-constant variance => Black–Scholes ---
    is_classic_cv = (sigma <= 1e-3 and abs(v0 - theta) <= 1e-8 and abs(rho) <= 1e-6 and kappa >= 5.0)
    is_degenerate = std2 < 1e-6  # tiny log-variance ⇒ COS interval collapses
    if is_classic_cv or is_degenerate:
        iv = np.sqrt(max(theta, 0.0))
        if option == "call":
            return float(black_scholes_price(S0, K, T, r, iv, option_type="call"))
        else:
            return float(black_scholes_price(S0, K, T, r, iv, option_type="put"))

    # --- Strike-centered truncation on y = ln(S_T) - ln(K), with safety guards ---
    c1_y = c1_x - np.log(K)
    std = np.sqrt(max(1e-8, std2))
    a = c1_y - L * std
    b = c1_y + L * std

    # Ensure 0 ∈ [a, b] so the exercise region (y >= 0) exists
    if a > 0.0:
        a = -1e-6
    if b < 0.0:
        b =  1e-6

    # Ensure a minimum width to avoid numerical degeneracy
    if (b - a) < 1e-3:
        mid = 0.5 * (a + b)
        a, b = mid - 5e-4, mid + 5e-4

    # --- COS expansion ---
    k = np.arange(int(N))
    u = k * np.pi / (b - a)

    # phi_y(u) = e^{-i u ln K} * phi_x(u); shift by 'a' for COS
    phi_x = heston_charfunc(u, T, r, kappa, theta, sigma, v0, rho, S0=S0)
    phi_y = phi_x * np.exp(-1j * u * np.log(K)) * np.exp(-1j * u * a)

    # Payoff coefficients for G(y) = (e^y - 1)^+ over [a, b]
    Vk = _cos_coefficients_call_y(a, b, k)
    Vk[0] *= 0.5  # first term has weight 1/2

    price_call = np.exp(-r*T) * K * np.real(np.sum(phi_y * Vk))

    if option == "call":
        return float(price_call)
    elif option == "put":
        return float(price_call - S0 + K * np.exp(-r*T))  # put via parity
    else:
        raise ValueError("option must be 'call' or 'put'")


def heston_call_put(S0, K, T, r, kappa, theta, sigma, v0, rho, N=4096, L=12, **kwargs):
    c = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="call", N=N, L=L, **kwargs)
    p = c - S0 + K * np.exp(-r*T)
    return c, p
