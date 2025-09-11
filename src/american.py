# options_pricing/american.py
# American option pricing via:
#   1) Longstaff–Schwartz Monte Carlo (LSMC)
#   2) Binomial Tree wrapper (reuses src.binomial_tree)

import numpy as np
from src.binomial_tree import binomial_tree

__all__ = [
    "american_put_lsmc",
    "american_call_lsmc",
    "american_lsmc",
    "american_binomial",
    "american_price",
]


def _simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed=None):
    """Simulate GBM paths under risk-neutral measure.
    Returns array of shape (n_paths, n_steps+1), including S(0).
    """
    dt = T / float(n_steps)
    rng = np.random.default_rng(seed)

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = float(S0)

    if sigma <= 0.0:
        growth = np.exp(r * dt)
        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t - 1] * growth
        return S

    nudt = (r - 0.5 * sigma**2) * dt
    sigsdt = sigma * np.sqrt(dt)
    Z = rng.standard_normal((n_paths, n_steps))

    logS = np.log(S0) + np.cumsum(nudt + sigsdt * Z, axis=1)
    S[:, 1:] = np.exp(logS)
    return S


def _poly_design_matrix(x, degree=2):
    """Simple polynomial design matrix [1, x, x^2, ...]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def american_lsmc(S0, K, T, r, sigma,
                  option="put", n_paths=50000, n_steps=50,
                  poly_degree=2, seed=0):
    """Price an American option via Longstaff–Schwartz (LSMC)."""
    dt = T / float(n_steps)
    disc = np.exp(-r * dt)

    S = _simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed=seed)

    if option == "call":
        payoff = np.maximum(S - K, 0.0)
    elif option == "put":
        payoff = np.maximum(K - S, 0.0)
    else:
        raise ValueError("option must be 'call' or 'put'")

    cashflows = payoff[:, -1].copy()

    for t in range(n_steps - 1, 0, -1):
        cashflows *= disc
        itm = payoff[:, t] > 0
        if not np.any(itm):
            continue

        X = S[itm, t]
        Y = cashflows[itm]

        A = _poly_design_matrix(X, degree=poly_degree)
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        continuation = A @ beta

        exercise_now = payoff[itm, t] > continuation
        if np.any(exercise_now):
            idx = np.where(itm)[0][exercise_now]
            cashflows[idx] = payoff[itm, t][exercise_now]

    price = float(np.mean(cashflows) * disc)
    return price


def american_put_lsmc(S0, K, T, r, sigma,
                      n_paths=50000, n_steps=50, poly_degree=2, seed=0):
    """Convenience wrapper for American put via LSMC."""
    return american_lsmc(S0, K, T, r, sigma, option="put",
                         n_paths=n_paths, n_steps=n_steps,
                         poly_degree=poly_degree, seed=seed)


def american_call_lsmc(S0, K, T, r, sigma,
                       n_paths=50000, n_steps=50, poly_degree=2, seed=0):
    """Convenience wrapper for American call via LSMC."""
    return american_lsmc(S0, K, T, r, sigma, option="call",
                         n_paths=n_paths, n_steps=n_steps,
                         poly_degree=poly_degree, seed=seed)


def american_binomial(S0, K, T, r, sigma, steps=400, option="put"):
    """American option via CRR binomial tree."""
    return float(binomial_tree(S0, K, T, r, sigma,
                               steps=steps, option_type=option, american=True))


def american_price(S0, K, T, r, sigma,
                   method="lsmc", option="put", **kwargs):
    """Unified entry point for American pricing."""
    if method == "lsmc":
        return american_lsmc(S0, K, T, r, sigma, option=option, **kwargs)
    elif method == "binomial":
        steps = int(kwargs.pop("steps", 400))
        return american_binomial(S0, K, T, r, sigma, steps=steps, option=option)
    else:
        raise ValueError("method must be 'lsmc' or 'binomial'")
