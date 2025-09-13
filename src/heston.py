
# options_pricing/heston.py
# Heston stochastic volatility model: European option pricing
# - Characteristic function \phi(u)
# - Carr–Madan (1999) Fourier pricing with exponential damping (alpha)

import numpy as np

__all__ = ["heston_charfunc", "heston_price", "heston_call_put"]


def heston_charfunc(u, T, r, kappa, theta, sigma, v0, rho):
    """
    Risk-neutral characteristic function for log-price ln(S_T).
    Returns E[exp(i u ln S_T)] where S_T is the underlying at time T.
    Parameters use the "Heston 1993" convention.
    """
    i = 1j
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * i * u - b)**2 + (sigma**2) * (i*u + u**2))
    g = (b - rho * sigma * i * u - d) / (b - rho * sigma * i * u + d)

    exp_negdT = np.exp(-d * T)
    one_minus_g_exp = 1 - g * exp_negdT
    one_minus_g = 1 - g

    C = r * i * u * T + (a / (sigma**2)) * ((b - rho * sigma * i * u - d) * T - 2.0 * np.log(one_minus_g_exp / one_minus_g))
    D = ((b - rho * sigma * i * u - d) / (sigma**2)) * ((1 - exp_negdT) / one_minus_g_exp)

    return np.exp(C + D * v0)


def _psi(u, S0, K, T, r, kappa, theta, sigma, v0, rho, alpha):
    """
    Integrand for Carr–Madan. Uses phi(u - i (alpha + 1)).
    """
    i = 1j
    x = np.log(S0)
    u_shift = u - i * (alpha + 1.0)
    phi = heston_charfunc(u_shift, T, r, kappa, theta, sigma, v0, rho)
    numer = np.exp(i * u * x) * phi
    denom = (alpha**2 + alpha - u**2 + i * (2.0 * alpha + 1.0) * u)
    return np.exp(-i * u * np.log(K)) * numer / denom


def heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="call",
                 alpha=1.5, N=4096, umax=200.0):
    """
    European option price under Heston via Carr–Madan Fourier transform.
    """
    u = np.linspace(1e-12, umax, int(N))
    integrand = np.real(np.exp(-r * T) * _psi(u, S0, K, T, r, kappa, theta, sigma, v0, rho, alpha))
    val = (np.trapz(integrand, u) / np.pi)

    call = np.exp(-alpha * np.log(K)) * val
    if option == "call":
        return float(call)
    elif option == "put":
        return float(call - S0 + K * np.exp(-r * T))
    else:
        raise ValueError("option must be 'call' or 'put'")


def heston_call_put(S0, K, T, r, kappa, theta, sigma, v0, rho, alpha=1.5, N=4096, umax=200.0):
    c = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="call", alpha=alpha, N=N, umax=umax)
    p = c - S0 + K * np.exp(-r * T)
    return c, p
