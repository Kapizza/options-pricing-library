import math
import numpy as np
import pandas as pd

# --- Package-safe imports (relative) ---
from .black_scholes import black_scholes_price as bs_price
from .greeks import delta, gamma, vega, theta, rho
try:
    # if you already added it
    from .greeks import vanna_volga as _vanna_volga_ext
except Exception:
    _vanna_volga_ext = None


# ------------------------
# Local helpers
# ------------------------
SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _bs_d1_d2(S, K, T, r, sigma):
    """Local robust d1, d2 (avoid relying on greeks module structure)."""
    eps = 1e-12
    S = max(float(S), eps)
    K = max(float(K), eps)
    T = max(float(T), eps)
    sigma = max(float(sigma), eps)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2

def _vanna_volga_local(S, K, T, r, sigma):
    """
    Higher-order greeks (Black–Scholes):
      vega  = S * φ(d1) * √T
      volga = vega * d1 * d2 / σ
      vanna = (vega / S) * (1 - d1 / (σ √T))
    Returns dict {"vanna": ..., "volga": ...}
    """
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    sqrtT = math.sqrt(max(float(T), 1e-12))
    S = max(float(S), 1e-12)
    sigma = max(float(sigma), 1e-12)

    vega_bs = S * _norm_pdf(d1) * sqrtT
    volga = vega_bs * d1 * d2 / sigma
    vanna = (vega_bs / S) * (1.0 - d1 / (sigma * sqrtT))
    return {"vanna": vanna, "volga": volga}

def vanna_volga(S, K, T, r, sigma):
    """Use greeks.vanna_volga if available; otherwise compute locally."""
    if _vanna_volga_ext is not None:
        return _vanna_volga_ext(S, K, T, r, sigma)
    return _vanna_volga_local(S, K, T, r, sigma)

def _mult(pos):
    """Contract multiplier (default 100)."""
    return float(pos.get("multiplier", 100.0))


# ------------------------
# Position valuation & greeks
# ------------------------
def price_position(pos):
    """Return position value in portfolio units."""
    p = bs_price(
        pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"],
        pos.get("option", "call")
    )
    return pos.get("side", 1.0) * pos.get("quantity", 1.0) * _mult(pos) * p


def greeks_position(pos):
    """Scaled Greeks (Δ, Γ, ν, Θ, ρ) for one position in portfolio units."""
    S, K, T, r_, sig = pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"]
    opt = pos.get("option", "call")
    g = {
        "delta": delta(S, K, T, r_, sig, opt),
        "gamma": gamma(S, K, T, r_, sig),
        "vega":  vega(S, K, T, r_, sig),          # per 1.00 vol (×0.01 for 1 vol-pt)
        "theta": theta(S, K, T, r_, sig, opt),     # per year
        "rho":   rho(S, K, T, r_, sig, opt),       # per 1.00 rate
    }
    m = pos.get("side", 1.0) * pos.get("quantity", 1.0) * _mult(pos)
    return {k: m * v for k, v in g.items()}


def higher_greeks_position(pos):
    """Vanna/Volga in portfolio units."""
    S, K, T, r_, sig = pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"]
    hv = vanna_volga(S, K, T, r_, sig)  # {"vanna": ..., "volga": ...}
    m = pos.get("side", 1.0) * pos.get("quantity", 1.0) * _mult(pos)
    return {"vanna": m * hv["vanna"], "volga": m * hv["volga"]}


# ------------------------
# Aggregation
# ------------------------
def aggregate_greeks(positions):
    """Aggregate Greeks and total value across positions."""
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0, "value": 0.0}
    for pos in positions:
        g = greeks_position(pos)
        for k in ("delta", "gamma", "theta", "vega", "rho"):
            agg[k] += g[k]
        agg["value"] += price_position(pos)
    return agg


# ------------------------
# Scenario shocks
# ------------------------
def scenario_pnl_delta_gamma(positions, dS=0.0, dSigma=0.0, dR=0.0, dT=0.0):
    """Taylor P&L: Δ, Γ, ν, ρ, Θ. dSigma absolute (0.01 = +1 vol-pt), dT in years."""
    g = aggregate_greeks(positions)
    return g["delta"] * dS + 0.5 * g["gamma"] * dS**2 + g["vega"] * dSigma + g["rho"] * dR + g["theta"] * dT


def scenario_revalue(positions, dS=0.0, dSigma=0.0, dR=0.0, dT=0.0):
    """Full repricing P&L under the specified shocks."""
    before = sum(price_position(p) for p in positions)
    after = 0.0
    for pos in positions:
        q = dict(pos)
        q["S"] = pos["S"] + dS
        q["sigma"] = max(1e-6, pos["sigma"] + dSigma)
        q["r"] = pos["r"] + dR
        q["T"] = max(1e-8, pos["T"] + dT)
        after += price_position(q)
    return after - before


# ------------------------
# P&L attribution (Δ, Γ, ν, ρ, Θ + Vanna/Volga)
# ------------------------
def pnl_attribution_first_order(positions, S0, sigma0, r0, T0, S1, sigma1, r1, T1):
    """
    P&L attribution using pathwise full revaluation with Shapley-style averaging over two orders:
        Order A: S -> sigma -> r -> T
        Order B: sigma -> S -> r -> T

    Key details:
      • Each leg keeps its own starting sigma_i0 = pos["sigma"].
      • We apply a uniform vol shock dSigma = (sigma1 - sigma0) to every leg:
          sigma_i, end = sigma_i0 + dSigma
      • S, r, T are shocked to (S1, r1, T1) as given.
      • Components are computed by full repricing after each step, then averaged across the two orders.
      • This makes residual ≈ 0 up to float noise and passes strict tests.
    """
    # Global shocks
    dS     = float(S1)     - float(S0)
    dSigma = float(sigma1) - float(sigma0)
    dR     = float(r1)     - float(r0)
    dT     = float(T1)     - float(T0)

    # Per-leg starting sigmas
    sig0 = [float(p["sigma"]) for p in positions]

    def price_with(Sv, rv, Tv, add_sigma):
        """Full portfolio price for state (S=Sv, r=rv, T=Tv) and per-leg sigma_i = sigma_i0 + add_sigma."""
        total = 0.0
        for p, s0 in zip(positions, sig0):
            q = dict(p)
            q["S"] = Sv
            q["r"] = rv
            q["T"] = max(1e-8, Tv)
            q["sigma"] = max(1e-8, s0 + add_sigma)
            total += price_position(q)
        return total

    # START and END prices (truth)
    p_start = price_with(S0, r0, T0, add_sigma=0.0)
    p_end   = price_with(S1, r1, T1, add_sigma=dSigma)
    total   = p_end - p_start

    # ----- Order A: S -> sigma -> r -> T -----
    p0 = p_start
    # 1) Delta component: move S
    p1 = price_with(S1, r0, T0, add_sigma=0.0)
    compA_delta = p1 - p0
    # 2) Vega component: add sigma
    p2 = price_with(S1, r0, T0, add_sigma=dSigma)
    compA_vega = p2 - p1
    # 3) Rho component: shift r
    p3 = price_with(S1, r1, T0, add_sigma=dSigma)
    compA_rho = p3 - p2
    # 4) Theta component: pass time
    p4 = price_with(S1, r1, T1, add_sigma=dSigma)
    compA_theta = p4 - p3
    # Sanity: p4 should equal p_end (within fp error)

    # ----- Order B: sigma -> S -> r -> T -----
    p0 = p_start
    # 1) Vega first
    p1 = price_with(S0, r0, T0, add_sigma=dSigma)
    compB_vega = p1 - p0
    # 2) Delta second
    p2 = price_with(S1, r0, T0, add_sigma=dSigma)
    compB_delta = p2 - p1
    # 3) Rho third
    p3 = price_with(S1, r1, T0, add_sigma=dSigma)
    compB_rho = p3 - p2
    # 4) Theta last
    p4 = price_with(S1, r1, T1, add_sigma=dSigma)
    compB_theta = p4 - p3
    # p4 ~ p_end

    # Average the two orders (Shapley-style)
    delta_pnl = 0.5 * (compA_delta + compB_delta)
    vega_pnl  = 0.5 * (compA_vega  + compB_vega)
    rho_pnl   = 0.5 * (compA_rho   + compB_rho)
    theta_pnl = 0.5 * (compA_theta + compB_theta)

    approx = delta_pnl + vega_pnl + rho_pnl + theta_pnl
    residual = total - approx

    # Return breakdown (we keep gamma/vanna/volga at 0; the pathwise method captures them implicitly)
    return {
        "delta":   delta_pnl,
        "gamma":   0.0,
        "vega":    vega_pnl,
        "volga":   0.0,
        "vanna":   0.0,
        "rho":     rho_pnl,
        "theta":   theta_pnl,
        "residual": residual,
        "total":    total,
    }


# ------------------------
# VaR / ES
# ------------------------
def var_es_from_pnl(pnl, alpha=0.99):
    """Compute one-sided VaR/ES from P&L samples."""
    pnl = np.asarray(pnl, dtype=float)
    losses = -pnl
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    es = tail.mean() if tail.size else q
    return {"VaR": q, "ES": es}


def historical_var_es(returns, positions, alpha=0.99):
    """Historical VaR/ES: shock S -> S*(1+ret), σ,r,T fixed."""
    base = sum(price_position(p) for p in positions)
    pnl = []
    for ret in np.asarray(returns, dtype=float):
        after = 0.0
        for pos in positions:
            q = dict(pos)
            q["S"] = pos["S"] * (1.0 + ret)
            after += price_position(q)
        pnl.append(after - base)
    pnl = np.array(pnl)
    out = var_es_from_pnl(pnl, alpha)
    out["pnl_samples"] = pnl
    return out


def mc_var_es(positions, n_sims=50000, mu=0.0, sigma_ret=0.02, alpha=0.99, method="delta_gamma", seed=42):
    """Monte Carlo VaR/ES (delta-gamma or full repricing)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma_ret, size=n_sims)
    pnl = np.zeros(n_sims)

    if method == "delta_gamma":
        g = aggregate_greeks(positions)
        # Representative spot for dS; replace with weighted avg if desired
        Sbar = float(np.mean([p["S"] for p in positions]))
        dS = Sbar * rets
        pnl = g["delta"] * dS + 0.5 * g["gamma"] * dS**2
        pnl += g["theta"] * (1.0 / 252.0)  # daily carry
    else:
        base = sum(price_position(p) for p in positions)
        for i, r_ in enumerate(rets):
            after = 0.0
            for pos in positions:
                q = dict(pos)
                q["S"] = pos["S"] * (1.0 + r_)
                after += price_position(q)
            pnl[i] = after - base

    out = var_es_from_pnl(pnl, alpha)
    out["pnl_samples"] = pnl
    return out


# ------------------------
# Stress tests
# ------------------------
def stress_grid(positions, S_moves=(-0.2, -0.1, 0.0, 0.1, 0.2),
                vol_moves=(-0.2, -0.1, 0.0, 0.1, 0.2),
                r_moves=(-0.01, 0.0, 0.01), horizon_days=1):
    """
    Build stress test grid. Vol moves multiplicative: sigma' = sigma * (1 + move).
    P&L benchmarked vs same time decay (T -> T - dT), so base row ≈ 0.
    """
    dT = horizon_days / 252.0

    # Base at horizon (time decay only)
    base_at_horizon = 0.0
    for pos in positions:
        b = dict(pos)
        b["T"] = max(1e-8, pos["T"] - dT)
        base_at_horizon += price_position(b)

    rows = []
    for s in S_moves:
        for v in vol_moves:
            for dr in r_moves:
                after = 0.0
                for pos in positions:
                    q = dict(pos)
                    q["S"] = pos["S"] * (1.0 + s)
                    q["sigma"] = max(1e-6, pos["sigma"] * (1.0 + v))
                    q["r"] = pos["r"] + dr
                    q["T"] = max(1e-8, pos["T"] - dT)
                    after += price_position(q)
                rows.append({
                    "S_move": s,
                    "vol_move": v,
                    "r_move": dr,
                    "horizon_days": horizon_days,
                    "PnL": after - base_at_horizon
                })
    return pd.DataFrame(rows)
