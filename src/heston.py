# options_pricing/heston.py
# Heston European option pricing via strike-centered COS (Fang & Oosterlee, 2008)
# with a safe Black–Scholes fallback in the near-constant-variance regime.

import numpy as np
from .black_scholes import black_scholes_price  # fallback in sigma→0 regime

__all__ = [
    "heston_charfunc",
    "heston_price",
    "heston_call_put",
    "heston_smile_prices",
]

def _log1p_complex(z):
    """
    Principal log(1 + z) for complex z, accurate for small |z|.
    np.log1p on complex input takes the real part as log(hypot(1 + Re z, Im z)),
    whose absolute error stays ~1e-16 however small z is.
    """
    zr, zi = np.real(z), np.imag(z)
    return 0.5 * np.log1p(zr * (2.0 + zr) + zi * zi) + 1j * np.arctan2(zi, 1.0 + zr)

def heston_charfunc(u, T, r, kappa, theta, sigma, v0, rho, S0=1.0):
    """
    Risk-neutral characteristic function of X_T = ln S_T.
    Returns E[exp(i u X_T)].

    Evaluated without cancellation. With beta = kappa - rho sigma i u and
    d = sqrt(beta^2 + sigma^2 (i u + u^2)), beta - d = -sigma^2 (i u + u^2)/(beta + d), so
        q = (beta - d)/sigma^2 = -(i u + u^2)/(beta + d),   g = (beta - d)/(beta + d) = sigma^2 q/(beta + d),
        log((1 - g e^{-dT})/(1 - g)) = log1p(-g e^{-dT}) - log1p(-g),
        C = i u (ln S0 + r T) + kappa theta [q T - 2 log((1 - g e^{-dT})/(1 - g))/sigma^2],
        D = q (1 - e^{-dT})/(1 - g e^{-dT}).
    The textbook form computes beta - d by subtraction and divides it by sigma^2,
    so its absolute error grows like 1/sigma^2 (3e-7 at sigma = 3e-5, T = 10).
    """
    i = 1j
    # At u = 0 with kappa = 0 the closed form is 0/0 (beta = d = 0); silence that
    # warning here and return the exact value phi(0) = E[1] = 1 below.
    with np.errstate(invalid="ignore"):
        iu = i * u
        beta = kappa - rho * sigma * iu
        w = iu + u * u
        d = np.sqrt(beta * beta + sigma**2 * w)
        q = -w / (beta + d)                 # (beta - d)/sigma^2
        g = sigma**2 * q / (beta + d)       # (beta - d)/(beta + d)
        exp_negdT = np.exp(-d * T)
        log_ratio = _log1p_complex(-g * exp_negdT) - _log1p_complex(-g)
        C = iu * (np.log(S0) + r * T) + kappa * theta * (q * T - 2.0 * log_ratio / sigma**2)
        D = q * (-np.expm1(-d * T)) / (1.0 - g * exp_negdT)
        phi = np.exp(C + D * v0)
    return np.where(u == 0, 1.0, phi)[()]

def _mean_integrated_variance(T, kappa, theta, v0):
    """E[int_0^T v_t dt] under Heston: theta*T + (v0 - theta)*(1 - e^{-kappa T})/kappa."""
    if kappa * T < 1e-10:
        return v0 * T
    return theta * T + (v0 - theta) * (-np.expm1(-kappa * T)) / kappa

def _cumulants_x(T, r, kappa, theta, sigma, v0, rho, S0):
    """
    First two cumulants of X = ln S_T, used for the COS truncation range.

    c1 is exact: ln S0 + r T - E[int v dt] / 2.
    c2 estimates the variance of ln S_T from the characteristic function by a
    central second difference of log(phi) at u = 0 (step h = 1e-3, using
    log phi(0) = 0 exactly), so it carries an O(h^2) truncation error plus the
    CF's round-off divided by h^2. The closed-form c2 printed in Fang &
    Oosterlee (2008) is about 2% off for their own test parameters and breaks
    down as kappa -> 0.
    """
    c1 = np.log(S0) + r*T - 0.5 * _mean_integrated_variance(T, kappa, theta, v0)
    if sigma < 1e-6:
        # (near-)deterministic variance: Var(ln S_T) = E[int v dt]
        return float(c1), max(1e-12, float(_mean_integrated_variance(T, kappa, theta, v0)))
    h = 1e-3
    lp = np.log(heston_charfunc(np.array([-h, h]), T, r, kappa, theta, sigma, v0, rho, S0=S0))
    c2 = -float((lp[0] + lp[1]).real) / (h * h)
    if not np.isfinite(c2):
        raise ValueError(
            "Heston variance of ln S_T is not finite for these parameters "
            f"(T={T}, kappa={kappa}, theta={theta}, sigma={sigma}, v0={v0}, rho={rho})."
        )
    return float(c1), max(1e-12, c2)

def _cumulant4_x(T, kappa, theta, sigma, v0, rho):
    """
    Fourth cumulant of X = ln S_T, for the COS truncation range.

    With s = i u, log E[e^{s X}] = s (ln S0 + r T) + A(T) + B(T) v0, where
    B' = (s^2 - s)/2 + (rho sigma s - kappa) B + sigma^2 B^2 / 2 and A' = kappa theta B.
    Writing B = sum_j b_j s^j and A = sum_j a_j s^j gives
        b1' = -1/2 - kappa b1
        b2' =  1/2 + rho sigma b1 - kappa b2 + sigma^2 b1^2 / 2
        b3' =  rho sigma b2 - kappa b3 + sigma^2 b1 b2
        b4' =  rho sigma b3 - kappa b4 + sigma^2 (b1 b3 + b2^2 / 2)
        a4' =  kappa theta b4
    from zero, and c4 = 24 (a4 + b4 v0). Solved by RK4 (relative error ~1e-6);
    a finite difference of log phi is unreliable here, because for heavy tails
    the Taylor series of log phi converges only very close to u = 0.
    """
    rs, s2, k, kt = rho * sigma, sigma * sigma, kappa, kappa * theta
    n = max(32, int(np.ceil(8.0 * kappa * T)))
    h = T / n

    def f(b1, b2, b3, b4):
        return (-0.5 - k * b1,
                0.5 + rs * b1 - k * b2 + 0.5 * s2 * b1 * b1,
                rs * b2 - k * b3 + s2 * b1 * b2,
                rs * b3 - k * b4 + s2 * (b1 * b3 + 0.5 * b2 * b2))

    b1 = b2 = b3 = b4 = a4 = 0.0
    for _ in range(n):
        k1 = f(b1, b2, b3, b4)
        y2 = (b1 + 0.5*h*k1[0], b2 + 0.5*h*k1[1], b3 + 0.5*h*k1[2], b4 + 0.5*h*k1[3])
        k2 = f(*y2)
        y3 = (b1 + 0.5*h*k2[0], b2 + 0.5*h*k2[1], b3 + 0.5*h*k2[2], b4 + 0.5*h*k2[3])
        k3 = f(*y3)
        y4 = (b1 + h*k3[0], b2 + h*k3[1], b3 + h*k3[2], b4 + h*k3[3])
        k4 = f(*y4)
        a4 += kt * h * (b4 + 2.0*y2[3] + 2.0*y3[3] + y4[3]) / 6.0
        b1 += h * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]) / 6.0
        b2 += h * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]) / 6.0
        b3 += h * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]) / 6.0
        b4 += h * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3]) / 6.0
    return float(24.0 * (a4 + b4 * v0))

def _cos_coefficients_put_y(a, b, k):
    """
    COS coefficients for payoff G(y) = (1 - e^y)^+ on y ∈ [a, b].
    Handles k = 0 safely. Exercise region is y ∈ [a, min(0,b)], empty if a >= 0.
    The payoff is bounded by 1, so the coefficients carry no e^b factor (the
    call payoff (e^y - 1)^+ has one, which amplifies errors in the CF and
    drops the right tail beyond b); calls follow from put-call parity.
    """
    k = np.asarray(k, dtype=float)
    omega = k * np.pi / (b - a)

    c = a
    d = min(0.0, b)
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

    Vk = (2.0 / (b - a)) * (psi(d, c) - chi(d, c))
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
    y := ln(S_T / K). Price(put) = e^{-rT} * K * sum Re[phi_y(u_k) * Vk]
    where phi_y(u) = e^{-i u ln K} * phi_x(u), u_k = k*pi/(b-a), and Vk are the
    coefficients of (1 - e^y)^+. Calls follow from parity: C = P + S0 - K e^{-rT}.

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
    is_degenerate = std2 < 1e-6 or sigma < 1e-6  # tiny log-variance or no vol-of-vol
    if is_classic_cv or is_degenerate:
        # sigma -> 0: variance is deterministic, so BS with the time-averaged variance
        iv = np.sqrt(max(_mean_integrated_variance(T, kappa, theta, v0) / T, 0.0))
        if option == "call":
            return float(black_scholes_price(S0, K, T, r, iv, option_type="call"))
        else:
            return float(black_scholes_price(S0, K, T, r, iv, option_type="put"))

    # --- Strike-centered truncation on y = ln(S_T) - ln(K), with safety guards ---
    # Scale sqrt(c2 + sqrt(c4)) (Fang & Oosterlee 2008): the put payoff is close to
    # K in the left tail, and for fat-tailed Heston laws that tail reaches past c1 - L sqrt(c2)
    c1_y = c1_x - np.log(K)
    c4_x = _cumulant4_x(T, kappa, theta, sigma, v0, rho)
    std = np.sqrt(max(1e-8, std2) + np.sqrt(max(c4_x, 0.0)))
    a = c1_y - L * std
    b = c1_y + L * std

    # Ensure 0 ∈ [a, b] so the exercise boundary (y = 0) lies in the window
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

    # Payoff coefficients for G(y) = (1 - e^y)^+ over [a, b]
    Vk = _cos_coefficients_put_y(a, b, k)
    Vk[0] *= 0.5  # first term has weight 1/2

    price_put = np.exp(-r*T) * K * np.real(np.sum(phi_y * Vk))

    if option == "call":
        return float(price_put + S0 - K * np.exp(-r*T))  # call via parity
    elif option == "put":
        return float(price_put)
    else:
        raise ValueError("option must be 'call' or 'put'")


def heston_call_put(S0, K, T, r, kappa, theta, sigma, v0, rho, N=4096, L=12, **kwargs):
    c = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="call", N=N, L=L, **kwargs)
    p = c - S0 + K * np.exp(-r*T)
    return c, p


def heston_smile_prices(
    S0,
    r,
    q,
    T,
    strikes,
    *,
    kappa,
    theta,
    sigma,
    v0,
    rho,
    N: int = 1536,
    L: float = 12.0,
    option: str = "call",
):
    """
    Vectorized Heston smile pricer using strike-centered COS over y = ln(S_T) - ln(K).

    Supports continuous dividend yield via the transformation S0' = S0 * exp(-q T),
    which is equivalent to using drift r in the COS formula with S0'. Parity under
    dividends is c - p = S0*exp(-qT) - K*exp(-rT). Puts are expanded with the
    bounded payoff (1 - e^y)^+; calls follow from that parity.

    Parameters
    ----------
    S0 : float
    r : float
    q : float
    T : float
    strikes : array-like
    kappa, theta, sigma, v0, rho : Heston parameters
    N : int
        Number of COS terms (k = 0..N-1)
    L : float
        Truncation width multiplier
    option : str
        "call" or "put"

    Returns
    -------
    prices : np.ndarray, shape (len(strikes),)
    """
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    if T <= 0:
        if option == "call":
            return np.maximum(S0 - strikes, 0.0)
        else:
            return np.maximum(strikes - S0, 0.0)

    # Dividend handling via S0' and r (see docstring)
    S0_eff = float(S0) * np.exp(-float(q) * float(T))
    r = float(r)
    T = float(T)

    # Cumulants for X = ln S_T using S0_eff and drift r
    c1_x, c2_x = _cumulants_x(T, r, kappa, theta, sigma, v0, rho, S0_eff)
    std2 = float(abs(c2_x))

    # Fallback to BS in near-constant-variance regime (use average variance ~ theta)
    is_classic_cv = (sigma <= 1e-3 and abs(v0 - theta) <= 1e-8 and abs(rho) <= 1e-6 and kappa >= 5.0)
    is_degenerate = std2 < 1e-6 or sigma < 1e-6
    if is_classic_cv or is_degenerate:
        from .black_scholes import black_scholes_price
        iv = np.sqrt(max(_mean_integrated_variance(T, kappa, theta, v0) / T, 0.0))
        if option == "call":
            return np.array([black_scholes_price(S0_eff, K, T, r, iv, option_type="call") for K in strikes], dtype=float)
        else:
            return np.array([black_scholes_price(S0_eff, K, T, r, iv, option_type="put") for K in strikes], dtype=float)

    # Truncation scale sqrt(c2 + sqrt(c4)) (Fang & Oosterlee 2008), see heston_price
    c4_x = _cumulant4_x(T, kappa, theta, sigma, v0, rho)
    std = np.sqrt(max(1e-8, std2) + np.sqrt(max(c4_x, 0.0)))
    width = 2.0 * L * std  # (b - a), constant across strikes

    # Frequency grid
    k = np.arange(int(N))
    u = k * np.pi / width  # u[0] = 0
    den = 1.0 + u*u
    nz = u != 0.0

    # phi_x(u) once, and shared strike-independent phase shift
    phi_x = heston_charfunc(u, T, r, kappa, theta, sigma, v0, rho, S0=S0_eff)
    phase = np.exp(-1j * u * (c1_x - L*std))  # shared across strikes
    phi_shared = phi_x * phase  # shape (N,)

    # Per-strike a = c1_y - L*std with c1_y = c1_x - ln K
    lnK = np.log(np.maximum(strikes, 1e-300))
    a = (c1_x - lnK) - L*std  # shape (M,)
    # Put exercise region y in [a, min(0, b)] with b = a + width; relative to a it
    # is [0, delta], empty (delta = 0) if a >= 0
    delta = np.clip(-a, 0.0, width)
    exp_a = np.exp(np.minimum(a, 0.0))           # e^a wherever the region is non-empty
    exp_d = np.exp(np.minimum(0.0, a + width))   # e^{min(0, b)}

    # Broadcast to (M, N)
    u_row = u[None, :]
    den_row = den[None, :]
    delta_col = delta[:, None]
    sin_d = np.sin(u_row * delta_col)
    cos_d = np.cos(u_row * delta_col)

    # psi: handle u=0 via definition (d - c = delta)
    psi = np.empty_like(sin_d)
    psi[:, nz] = sin_d[:, nz] / u_row[:, nz]
    psi[:, ~nz] = delta_col

    # chi
    chi = (exp_d[:, None] * (cos_d + u_row * sin_d) - exp_a[:, None]) / den_row

    # Vk
    Vk = (2.0 / width) * (psi - chi)
    Vk[:, 0] *= 0.5  # k=0 term half-weight
    # Exercise region y <= 0 lies entirely below the window [a, a + width]
    Vk[a >= 0.0, :] = 0.0

    # Put price = DF * K * Re(sum_k phi_shared * Vk)
    DF = np.exp(-r * T)
    accum = np.real(Vk @ phi_shared)
    puts = DF * strikes * accum

    if option == "call":
        # call via parity with dividends: c = p + S0*e^{-qT} - K e^{-rT}
        return (puts + S0_eff - strikes * DF).astype(float)
    else:
        return puts.astype(float)
