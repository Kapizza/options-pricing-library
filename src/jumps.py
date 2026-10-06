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

def _cos_coeff_unit_put(a, b, N):
    # F_k for payoff max(1 - e^y, 0) with the integration domain [a, b], per strike:
    # a, b have shape (M,), the result (M, N). Exercise region [a, min(0, b)], empty if a >= 0.
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[:, None]
    k = np.arange(N)
    omega = k * math.pi / (b - a)

    def chi(xl, xu):
        c = np.cos(omega * (xu - a)) * np.exp(xu) - np.cos(omega * (xl - a)) * np.exp(xl)
        s = omega * (np.sin(omega * (xu - a)) * np.exp(xu) - np.sin(omega * (xl - a)) * np.exp(xl))
        return (c + s) / (1.0 + omega * omega)

    def psi(xl, xu):
        out = (np.sin(omega * (xu - a)) - np.sin(omega * (xl - a))) / np.where(omega == 0.0, 1.0, omega)
        out[:, 0] = (xu - xl)[:, 0]
        return out

    xl, xu = a, np.maximum(np.minimum(0.0, b), a)
    Fk = 2.0 / (b - a) * (psi(xl, xu) - chi(xl, xu))
    Fk[a[:, 0] >= 0.0, :] = 0.0
    return Fk


def _truncation_range_logmoneyness(T, r, q, sigma, lam, muJ, sigJ, L=12):
    """
    Truncation [a, b] = [c1 - L s, c1 + L s] for Y = log(S_T/S0), with the scale
    s = sqrt(c2 + sqrt(c4)) of Fang & Oosterlee (2008) for fat tails: rare large
    jumps put mass far beyond L sqrt(c2). The pricer shifts the window by
    x0 = log(S0/K) per strike, so that the window for y = log(S_T/K) = x0 + Y is
    centred on the mean of y.
    """
    kappa = math.exp(muJ + 0.5 * sigJ * sigJ) - 1.0
    # Cumulants of Y_T (the jump part contributes lam T E[J^n])
    c1 = (r - q - 0.5 * sigma * sigma - lam * kappa) * T + lam * T * muJ
    c2 = sigma * sigma * T + lam * T * (sigJ * sigJ + muJ * muJ)
    c4 = lam * T * (muJ ** 4 + 6.0 * muJ * muJ * sigJ * sigJ + 3.0 * sigJ ** 4)
    s = math.sqrt(max(c2 + math.sqrt(max(c4, 0.0)), 1e-16))
    a = c1 - L * s
    b = c1 + L * s
    return a, b


def merton_price_cos(S0, K, T, r, q, sigma, lam, muJ, sigJ,
                     option="call", N=2048, L=12, return_components=False):
    """
    COS pricing in y = log(S_T/K) = x0 + Y, x0 = log(S0/K), Y = log(S_T/S0).
    Each strike gets the window [a, b] = [x0 + c1 - L s, x0 + c1 + L s],
    s = sqrt(c2 + sqrt(c4)), centred on the mean of y; the width is shared, so
    u_k and φ_Y(u_k) are too.
    Puts are expanded with the bounded payoff (1 - e^y)^+:
        Put = K * e^{-rT} * sum_k Re[ φ_Y(u_k) * exp(i u_k (x0 - a)) ] * F_k,
    and calls follow from parity, C = P + S0 e^{-qT} - K e^{-rT} (the call payoff
    grows like e^y, so its coefficients carry e^b and the right tail beyond b is lost).
    With return_components, also returns (u, a, b, F) with per-strike a, b of
    shape (M,) and put coefficients F of shape (M, N).
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    K = np.atleast_1d(np.asarray(K, dtype=float))
    lo, hi = _truncation_range_logmoneyness(T, r, q, sigma, lam, muJ, sigJ, L=L)

    k = np.arange(N)
    u = k * math.pi / (hi - lo)
    phi = cf_merton(u, T, r, q, sigma, lam, muJ, sigJ)

    # Per-strike windows for y = x0 + Y; exp(i u (x0 - a)) = exp(-i u lo) is shared
    x0 = np.log(max(S0, 1e-300) / np.maximum(K, 1e-300))
    a = x0 + lo
    b = x0 + hi
    Fk = _cos_coeff_unit_put(a, b, N)

    # COS weights
    w = np.ones(N)
    w[0] = 0.5

    disc = math.exp(-r * T)
    puts = K * disc * (Fk @ (w * np.real(phi * np.exp(-1j * u * lo))))
    if option == "call":
        prices = puts + S0 * math.exp(-q * T) - K * disc
    else:
        prices = puts

    if prices.size == 1:
        prices = float(prices[0])

    if return_components:
        return prices, (u, a, b, Fk)
    return prices




# Convenience: parity and sanity checks helpers

def merton_call_put_parity(S0, K, T, r, q, price_call):
    """Return the implied put from call via parity under any model."""
    return price_call - S0 * math.exp(-q * T) + K * math.exp(-r * T)


__all__ = [
    "cf_merton",
    "merton_price_cos",
    "merton_call_put_parity",
]
