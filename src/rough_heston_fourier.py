# options_pricing/rough_heston_fourier.py
# Semi-analytic rough Heston pricing: characteristic function from the
# fractional Riccati equation (El Euch & Rosenbaum 2019), solved with the
# fractional Adams scheme (Diethelm, Ford & Freed 2002) with an implicit
# corrector, and
# European prices by the COS method (Fang & Oosterlee 2008).
#
# Model (same parameterisation as src.rough.rough_heston_paths):
#   V_t = v0 + 1/Gamma(alpha) int_0^t (t-s)^(alpha-1) [kappa(theta - V_s) ds
#                                                     + eta sqrt(V_s) dW_s],
#   alpha = H + 1/2, d ln S = (r - q - V/2) dt + sqrt(V) dB, d<W,B> = rho dt.
# For H = 1/2 this is the classical Heston model.

import math
import numpy as np

__all__ = [
    "rough_heston_charfunc",
    "rough_heston_price",
    "rough_heston_smile_prices",
]


def _adams_weights(alpha, n):
    """Corrector weights of the fractional Adams scheme on n steps (dimensionless)."""
    # corrector: interior weights c_m = (m+1)^(a+1) + (m-1)^(a+1) - 2 m^(a+1), m >= 1
    c = np.zeros(n + 2)
    m = np.arange(1, n + 2, dtype=float)
    c[1:] = (m + 1.0) ** (alpha + 1.0) + (m - 1.0) ** (alpha + 1.0) - 2.0 * m ** (alpha + 1.0)
    return c


def _fractional_integral_end(f, beta, dt):
    """
    I^beta f at the last grid point, from values f[..., 0..n] on a uniform grid,
    by product-trapezoidal weights (exact for piecewise-linear f). beta = 0
    returns f at the last point.
    """
    n = f.shape[-1] - 1
    if beta == 0.0:
        return f[..., -1]
    k = np.arange(n + 1, dtype=float)
    w = np.empty(n + 1)
    w[0] = (n - 1.0) ** (beta + 1.0) - (n - 1.0 - beta) * n ** beta if n >= 1 else 1.0
    jj = k[1:n]
    w[1:n] = ((n - jj + 1.0) ** (beta + 1.0) + (n - jj - 1.0) ** (beta + 1.0)
              - 2.0 * (n - jj) ** (beta + 1.0))
    w[n] = 1.0
    return dt ** beta / math.gamma(beta + 2.0) * (f @ w)


def _solve_fractional_riccati(u, T, kappa, eta, rho, H, n_steps):
    """
    h(u, t) on t_k = k T / n for D^alpha h = F(u, h), h(0) = 0, with
    F(u, x) = (-u^2 - i u)/2 + (i u rho eta - kappa) x + eta^2 x^2 / 2.
    Returns an array of shape (len(u), n_steps + 1).
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    alpha = H + 0.5
    n = int(n_steps)
    dt = T / n
    c0 = 0.5 * (-u * u - 1j * u)
    c1 = 1j * u * rho * eta - kappa
    c2 = 0.5 * eta * eta

    def F(x):
        return c0 + c1 * x + c2 * x * x

    c = _adams_weights(alpha, n)
    corr = dt ** alpha / math.gamma(alpha + 2.0)

    h = np.zeros((u.size, n + 1), dtype=complex)
    Fh = np.zeros((u.size, n + 1), dtype=complex)
    Fh[:, 0] = F(h[:, 0])
    # The corrector h_{k+1} = corr (F(h_{k+1}) + acc) is solved implicitly: it is a
    # quadratic in h_{k+1}. For large |u| the equation is stiff (F ~ -u^2/2 and
    # (i u rho eta - kappa) h) and the explicit predictor-corrector diverges unless
    # dt^alpha |u| is small; the implicit step stays stable. Of the two roots we
    # take the one nearest the explicit predictor-corrector value (it tends to the
    # linear solution as eta -> 0).
    qa = corr * c2
    qb = corr * c1 - 1.0
    for k in range(n):
        a0 = k ** (alpha + 1.0) - (k - alpha) * (k + 1.0) ** alpha
        acc = a0 * Fh[:, 0]
        if k >= 1:
            acc = acc + Fh[:, 1:k + 1] @ c[k:0:-1]
        qc = corr * (c0 + acc)
        disc = np.sqrt(qb * qb - 4.0 * qa * qc)
        # numerically stable roots of qa x^2 + qb x + qc = 0
        s = -0.5 * (qb + np.where((np.conj(qb) * disc).real >= 0.0, disc, -disc))
        r1 = qc / s
        r2 = s / qa if qa != 0.0 else r1
        x_lin = -qc / qb
        h[:, k + 1] = np.where(np.abs(r1 - x_lin) <= np.abs(r2 - x_lin), r1, r2)
        Fh[:, k + 1] = F(h[:, k + 1])
    return h


def rough_heston_charfunc(u, T, r, kappa, theta, eta, v0, rho, H, S0=1.0, q=0.0, n_steps=200):
    """
    Characteristic function E[exp(i u ln S_T)] of the rough Heston model:
        exp(i u (ln S0 + (r - q) T) + kappa theta I^1 h(T) + v0 I^(1-alpha) h(T)),
    with h the solution of the fractional Riccati equation (El Euch & Rosenbaum
    2019, Theorem 2.1) and alpha = H + 1/2. At H = 1/2 it is the classical
    Heston characteristic function. n_steps sets the time grid of the Adams
    scheme (error O(T/n_steps)^(1+alpha) in h for smooth solutions).
    """
    scalar = np.ndim(u) == 0
    u = np.asarray(u, dtype=float).reshape(-1)
    T = float(T)
    alpha = H + 0.5
    if T <= 0.0:
        out = np.exp(1j * u * math.log(S0))
        return out[0] if scalar else out
    n = int(n_steps)
    dt = T / n
    h = _solve_fractional_riccati(u, T, kappa, eta, rho, H, n)
    int_h = dt * (h[:, 1:-1].sum(axis=1) + 0.5 * (h[:, 0] + h[:, -1]))    # I^1 h(T)
    frac_h = _fractional_integral_end(h, 1.0 - alpha, dt)                 # I^(1-alpha) h(T)
    out = np.exp(1j * u * (math.log(S0) + (r - q) * T) + kappa * theta * int_h + v0 * frac_h)
    out = np.where(u == 0.0, 1.0 + 0.0j, out)
    return out[0] if scalar else out


def _cumulants(T, r, kappa, theta, eta, v0, rho, H, S0, q, n_steps):
    """c1, c2 and c4 of ln S_T from finite differences of log phi (step h)."""
    hs = 0.05 / math.sqrt(max(T * max(v0, theta, 1e-4), 1e-8))
    hs = min(hs, 1.0)
    uu = np.array([-2.0 * hs, -hs, hs, 2.0 * hs])
    # Cumulants of ln(S_T/S0) - (r-q)T: its phase is small, so the complex log
    # does not wrap (with ln S0 included, h ln S0 can exceed pi for short T).
    lp = np.log(rough_heston_charfunc(uu, T, 0.0, kappa, theta, eta, v0, rho, H,
                                      S0=1.0, q=0.0, n_steps=n_steps))
    lm2, lm1, lp1, lp2 = lp
    c1 = math.log(S0) + (r - q) * T + float(((lp1 - lm1) / (2.0 * hs)).imag)
    # Re log phi(u) = -c2 u^2/2 + c4 u^4/24 - ...: fit from u = h and 2h
    e1 = float((lp1 + lm1).real) / 2.0
    e2 = float((lp2 + lm2).real) / 2.0
    c2 = -(16.0 * e1 - e2) / (6.0 * hs * hs)          # eliminates the u^4 term
    c4 = 2.0 * (e2 - 4.0 * e1) / hs ** 4               # eliminates the u^2 term
    if not np.isfinite(c2) or not np.isfinite(c1):
        raise ValueError("rough Heston cumulants are not finite for these parameters")
    return c1, max(c2, 1e-12), max(c4, 0.0)


def rough_heston_smile_prices(S0, r, q, T, strikes, *, kappa, theta, eta, v0, rho, H,
                              N=512, L=12.0, n_steps=200, option="call"):
    """
    European prices under rough Heston for a vector of strikes, by COS on
    y = ln(S_T / K). Puts use the bounded put payoff and calls follow from
    put-call parity c - p = S0 e^{-qT} - K e^{-rT}; the truncation range is
    c1 +- L sqrt(c2 + sqrt(c4)) (Fang & Oosterlee 2008).
    """
    strikes = np.asarray(strikes, dtype=float).reshape(-1)
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if T <= 0:
        return np.maximum(S0 - strikes, 0.0) if option == "call" else np.maximum(strikes - S0, 0.0)
    T = float(T)
    c1, c2, c4 = _cumulants(T, r, kappa, theta, eta, v0, rho, H, S0, q, n_steps)
    std = math.sqrt(c2 + math.sqrt(c4))
    width = 2.0 * L * std

    k = np.arange(int(N))
    u = k * np.pi / width
    den = 1.0 + u * u
    nz = u != 0.0
    phi = rough_heston_charfunc(u, T, r, kappa, theta, eta, v0, rho, H, S0=S0, q=q, n_steps=n_steps)
    phi_shared = phi * np.exp(-1j * u * (c1 - L * std))

    lnK = np.log(np.maximum(strikes, 1e-300))
    a = (c1 - lnK) - L * std
    delta = np.clip(-a, 0.0, width)
    exp_a = np.exp(np.minimum(a, 0.0))
    exp_d = np.exp(np.minimum(0.0, a + width))
    u_row = u[None, :]
    delta_col = delta[:, None]
    sin_d = np.sin(u_row * delta_col)
    cos_d = np.cos(u_row * delta_col)
    psi = np.empty_like(sin_d)
    psi[:, nz] = sin_d[:, nz] / u_row[:, nz]
    psi[:, ~nz] = delta_col
    chi = (exp_d[:, None] * (cos_d + u_row * sin_d) - exp_a[:, None]) / den[None, :]
    Vk = (2.0 / width) * (psi - chi)
    Vk[:, 0] *= 0.5
    Vk[a >= 0.0, :] = 0.0

    DF = math.exp(-r * T)
    puts = DF * strikes * np.real(Vk @ phi_shared)
    if option == "put":
        return puts
    return puts + S0 * math.exp(-q * T) - strikes * DF


def rough_heston_price(S0, K, T, r, kappa, theta, eta, v0, rho, H,
                       option="call", q=0.0, N=512, L=12.0, n_steps=200):
    """European call or put under rough Heston (see rough_heston_smile_prices)."""
    return float(rough_heston_smile_prices(S0, r, q, T, [K], kappa=kappa, theta=theta, eta=eta,
                                           v0=v0, rho=rho, H=H, N=N, L=L, n_steps=n_steps,
                                           option=option)[0])
