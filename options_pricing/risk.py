
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def _bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        eps = 1e-12
        T = max(T, eps)
        sigma = max(sigma, eps)
        S = max(S, eps)
        K = max(K, eps)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> float:
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    if option.lower() == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option: str = "call") -> Dict[str, float]:
    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    pdf = _norm_pdf(d1)
    if option.lower() == "call":
        delta = _norm_cdf(d1)
        theta = -(S * pdf * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _norm_cdf(d2)
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = -(S * pdf * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

def _mult(pos: Dict) -> float:
    return float(pos.get("multiplier", 100.0))

def price_position(pos: Dict) -> float:
    p = bs_price(pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"], pos.get("option", "call"))
    return pos.get("side", 1.0) * pos.get("quantity", 1.0) * _mult(pos) * p

def greeks_position(pos: Dict) -> Dict[str, float]:
    g = bs_greeks(pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"], pos.get("option", "call"))
    m = pos.get("side", 1.0) * pos.get("quantity", 1.0) * _mult(pos)
    return {k: m * v for k, v in g.items()}

def aggregate_greeks(positions: List[Dict]) -> Dict[str, float]:
    agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0, "value": 0.0}
    for pos in positions:
        g = greeks_position(pos)
        for k in ("delta", "gamma", "theta", "vega", "rho"):
            agg[k] += g[k]
        agg["value"] += price_position(pos)
    return agg

def scenario_pnl_delta_gamma(
    positions: List[Dict],
    dS: float = 0.0,
    dSigma: float = 0.0,
    dR: float = 0.0,
    dT: float = 0.0,
) -> float:
    g = aggregate_greeks(positions)
    pnl = g["delta"] * dS + 0.5 * g["gamma"] * dS * dS + g["vega"] * dSigma + g["rho"] * dR + g["theta"] * dT
    return pnl

def scenario_revalue(positions: List[Dict], dS: float=0.0, dSigma: float=0.0, dR: float=0.0, dT: float=0.0) -> float:
    before = 0.0
    after = 0.0
    for pos in positions:
        before += price_position(pos)
        pos2 = dict(pos)
        pos2["S"] = pos["S"] + dS
        pos2["sigma"] = max(1e-6, pos["sigma"] + dSigma)
        pos2["r"] = pos["r"] + dR
        pos2["T"] = max(1e-8, pos["T"] + dT)
        after += price_position(pos2)
    return after - before

def pnl_attribution_first_order(
    positions: List[Dict],
    S0: float, sigma0: float, r0: float, T0: float,
    S1: float, sigma1: float, r1: float, T1: float
) -> Dict[str, float]:
    shadow = []
    for pos in positions:
        p = dict(pos)
        p["S"], p["sigma"], p["r"], p["T"] = S0, sigma0, r0, T0
        shadow.append(p)
    g0 = aggregate_greeks(shadow)

    dS = S1 - S0
    dSigma = sigma1 - sigma0
    dR = r1 - r0
    dT = T1 - T0

    approx = g0["delta"] * dS + g0["vega"] * dSigma + g0["rho"] * dR + g0["theta"] * dT

    before = sum(price_position(p) for p in shadow)
    end_positions = []
    for pos in shadow:
        q = dict(pos)
        q["S"], q["sigma"], q["r"], q["T"] = S1, sigma1, r1, T1
        end_positions.append(q)
    after = sum(price_position(p) for p in end_positions)

    total = after - before
    residual = total - approx
    return {"delta": g0["delta"] * dS, "vega": g0["vega"] * dSigma, "rho": g0["rho"] * dR, "theta": g0["theta"] * dT,
            "residual": residual, "total": total}

def var_es_from_pnl(pnl: np.ndarray, alpha: float = 0.99) -> Dict[str, float]:
    pnl = np.asarray(pnl).astype(float)
    losses = -pnl
    q = np.quantile(losses, alpha)
    tail = losses[losses >= q]
    es = tail.mean() if tail.size else q
    return {"VaR": q, "ES": es}

def historical_var_es(returns: np.ndarray, positions: List[Dict], alpha: float = 0.99) -> Dict[str, float]:
    pnl = []
    for ret in returns:
        shocked = []
        for pos in positions:
            p = dict(pos)
            p["S"] = pos["S"] * (1.0 + ret)
            shocked.append(p)
        diff = sum(price_position(p) for p in shocked) - sum(price_position(p) for p in positions)
        pnl.append(diff)
    pnl = np.array(pnl)
    out = var_es_from_pnl(pnl, alpha=alpha)
    out["pnl_samples"] = pnl
    return out

def mc_var_es(
    positions: List[Dict],
    n_sims: int = 50_000,
    mu: float = 0.0,
    sigma_ret: float = 0.02,
    alpha: float = 0.99,
    method: str = "delta_gamma",
    seed: Optional[int] = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma_ret, size=n_sims)
    pnl = np.zeros(n_sims, dtype=float)

    if method == "delta_gamma":
        g = aggregate_greeks(positions)
        Sbar = positions[0]["S"]
        dS = Sbar * rets
        pnl = g["delta"] * dS + 0.5 * g["gamma"] * dS * dS
        pnl += g["theta"] * (1.0/252.0)
    else:
        base = sum(price_position(p) for p in positions)
        for i, r in enumerate(rets):
            shocked = []
            for pos in positions:
                q = dict(pos)
                q["S"] = pos["S"] * (1.0 + r)
                shocked.append(q)
            pnl[i] = sum(price_position(q) for q in shocked) - base

    out = var_es_from_pnl(pnl, alpha=alpha)
    out["pnl_samples"] = pnl
    return out

def stress_grid(
    positions: List[Dict],
    S_moves: List[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
    vol_moves: List[float] = (-0.2, -0.1, 0.0, 0.1, 0.2),
    r_moves: List[float] = (-0.01, 0.0, 0.01),
    horizon_days: int = 1
) -> pd.DataFrame:
    rows = []
    base_val = sum(price_position(p) for p in positions)
    for s in S_moves:
        for v in vol_moves:
            for dr in r_moves:
                dT = horizon_days / 252.0
                after = 0.0
                for pos in positions:
                    q = dict(pos)
                    q["S"] = pos["S"] * (1.0 + s)
                    q["sigma"] = max(1e-6, pos["sigma"] * (1.0 + v))
                    q["r"] = pos["r"] + dr
                    q["T"] = max(1e-8, pos["T"] - dT)
                    after += price_position(q)
                rows.append({
                    "S_move": s, "vol_move": v, "r_move": dr, "horizon_days": horizon_days,
                    "PnL": after - base_val
                })
    df = pd.DataFrame(rows)
    return df
