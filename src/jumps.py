"""
Jump-diffusion pricing via the COS method.

Implements Merton (1976) jump-diffusion using a fast and stable
Fourier-COS expansion. The interface mirrors the rest of the
library and supports vectorized strikes.

Design
- Dependency-light (numpy + math)
- No numpy.typing and no typing imports
- Batch pricing across strikes using shared CF evaluations

"""
import math
import numpy as np

# -------------------------
# Characteristic functions
# -------------------------

def cf_merton(u, T, r, q, sigma, lam, muJ, sigJ):
    """Characteristic function of log S_T under Merton JD.

    Let X_T = log S_T. Under the risk-neutral measure with dividend yield q,
    the drift of log S is (r - q - 0.5*sigma^2 - lam*kappa), where
    kappa = E[J-1] = exp(muJ + 0.5*sigJ^2) - 1.

    Parameters
    ----------
    u : array-like
        Fourier argument(s).
    T : float
        Maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Diffusive volatility.
    lam : float
        Jump intensity (Poisson rate).
    muJ : float
        Mean of jump size in log space (log-normal jump size).
    sigJ : float
        Std dev of jump size in log space.
    """

    u = np.asarray(u, dtype=float)
    iu = 1j * u
    kappa = math.exp(muJ + 0.5 * sigJ * sigJ) - 1.0
    drift = (r - q - 0.5 * sigma * sigma - lam * kappa)
    diff_cf = np.exp(iu * drift * T - 0.5 * sigma * sigma * u * u * T)
    jump_cf = np.exp(lam * T * (np.exp(iu * muJ - 0.5 * sigJ * sigJ * u * u) - 1.0))
    return diff_cf * jump_cf


# -------------------------
# COS helper utilities
# -------------------------

def _cos_coeff_unit_call(a, b, N):
    # F_k for payoff max(e^y - 1, 0) with the integration domain [a, b]
    k = np.arange(N)
    omega = k * math.pi / (b - a)

    def chi(xl, xu):
        # ∫_xl^xu e^y cos(ω (y - a)) dy
        c = np.cos(omega * (xu - a)) * np.exp(xu) - np.cos(omega * (xl - a)) * np.exp(xl)
        s = omega * (np.sin(omega * (xu - a)) * np.exp(xu) - np.sin(omega * (xl - a)) * np.exp(xl))
        return (c + s) / (1.0 + omega * omega)

    def psi(xl, xu):
        # ∫_xl^xu cos(ω (y - a)) dy, with k=0 limit handled
        out = (np.sin(omega * (xu - a)) - np.sin(omega * (xl - a))) / np.where(omega == 0.0, 1.0, omega)
        out[0] = (xu - xl)
        return out

    xl, xu = 0.0, b
    Fk = 2.0 / (b - a) * (chi(xl, xu) - psi(xl, xu))
    return Fk


def _cos_coeff_unit_put(a, b, N):
    # F_k for payoff max(1 - e^y, 0) with the integration domain [a, b]
    k = np.arange(N)
    omega = k * math.pi / (b - a)

    def chi(xl, xu):
        c = np.cos(omega * (xu - a)) * np.exp(xu) - np.cos(omega * (xl - a)) * np.exp(xl)
        s = omega * (np.sin(omega * (xu - a)) * np.exp(xu) - np.sin(omega * (xl - a)) * np.exp(xl))
        return (c + s) / (1.0 + omega * omega)

    def psi(xl, xu):
        out = (np.sin(omega * (xu - a)) - np.sin(omega * (xl - a))) / np.where(omega == 0.0, 1.0, omega)
        out[0] = (xu - xl)
        return out

    xl, xu = a, 0.0
    Fk = 2.0 / (b - a) * (psi(xl, xu) - chi(xl, xu))
    return Fk


def _truncation_range_logmoneyness(T, r, q, sigma, lam, muJ, sigJ, L=12):
    """
    Truncation [a, b] for y = log(S_T/K) = log(S_T/S0) + log(S0/K).
    Because we price with the CF of Y = log(S_T/S0) and later shift by x0 = log(S0/K),
    we center [a, b] on Y's cumulants (independent of K, S0).
    """
    kappa = math.exp(muJ + 0.5 * sigJ * sigJ) - 1.0
    # Mean and variance of Y_T
    c1 = (r - q - 0.5 * sigma * sigma - lam * kappa) * T + lam * T * muJ
    c2 = sigma * sigma * T + lam * T * (sigJ * sigJ + muJ * muJ)
    a = c1 - L * math.sqrt(max(c2, 1e-16))
    b = c1 + L * math.sqrt(max(c2, 1e-16))
    return a, b


def merton_price_cos(S0, K, T, r, q, sigma, lam, muJ, sigJ,
                     option="call", N=2048, L=12, return_components=False):
    """
    COS pricing in y = log(S_T/K). Price = K * e^{-rT} * sum_k Re[ φ_Y(u_k) * exp(i u_k (x0 - a)) ] * F_k
    where x0 = log(S0/K), φ_Y is CF of log-return, and F_k are unit-payoff coefficients.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    a, b = _truncation_range_logmoneyness(T, r, q, sigma, lam, muJ, sigJ, L=L)

    k = np.arange(N)
    u = k * math.pi / (b - a)
    phi = cf_merton(u, T, r, q, sigma, lam, muJ, sigJ)

    if option == "call":
        Fk = _cos_coeff_unit_call(a, b, N)
    elif option == "put":
        Fk = _cos_coeff_unit_put(a, b, N)
    else:
        raise ValueError("option must be 'call' or 'put'")

    # COS weights
    w = np.ones(N)
    w[0] = 0.5

    disc = math.exp(-r * T)
    prices = np.empty_like(K)

    # For each strike, shift by x0 = log(S0/K)
    for i, Ki in enumerate(K):
        x0 = math.log(max(S0, 1e-300) / max(Ki, 1e-300))
        phase = np.exp(1j * u * (x0 - a))
        series = w * Fk * np.real(phi * phase)
        prices[i] = Ki * disc * np.sum(series)

    if prices.size == 1:
        prices = float(prices[0])

    if return_components:
        return prices, (u, a, b, Fk)
    return prices




def merton_call_put_parity(S0, K, T, r, q, price_call):
    return price_call - S0 * math.exp(-q * T) + K * math.exp(-r * T)


# Convenience: parity and sanity checks helpers

def merton_call_put_parity(S0, K, T, r, q, price_call):
    """Return the implied put from call via parity under any model."""
    return price_call - S0 * math.exp(-q * T) + K * math.exp(-r * T)


__all__ = [
    "cf_merton",
    "merton_price_cos",
    "merton_call_put_parity",
]
