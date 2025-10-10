# src/calibration_rough.py
# Calibration helpers for rough Heston and rBergomi
# Dependencies: numpy, scipy (optimize), your existing src modules:
#   - src.rough_heston.rough_heston_euro_mc, rough_heston_paths
#   - src.rough.rbergomi_paths
#   - src.black_scholes.black_scholes_price
#   - src.volatility.implied_vol  (or fall back to a local brent if needed)

import numpy as np
import math

from scipy.optimize import minimize, Bounds

from src.black_scholes import black_scholes_price
from src.volatility import implied_volatility as _iv_solve
from src.greeks import vega as bs_vega


import time
import numpy as np

class _CalibMonitor:
    def __init__(self, tag="calib", print_every=1, verbose=True):
        self.tag = tag
        self.print_every = max(1, int(print_every))
        self.verbose = bool(verbose)
        self.iter = 0
        self.last_x = None
        self.last_f = None
        self.t0 = None
        self.t_last = None
        self.history = []  # list of dicts

    def wrap_obj(self, f):
        # Wrap objective to record (x, f) without extra evals
        def _wrapped(x):
            val = f(x)
            self.last_x = np.array(x, dtype=float)
            self.last_f = float(val)
            return val
        return _wrapped

    def start(self, start_idx=1):
        self.iter = 0
        self.last_x = None
        self.last_f = None
        self.t0 = time.time()
        self.t_last = self.t0
        if self.verbose:
            print(f"[{self.tag} start #{start_idx}] iter=0")

    def cb(self, xk):
        # Called by scipy after each iteration
        self.iter += 1
        now = time.time()
        dt = now - self.t_last
        self.t_last = now

        # deltas if available
        df = np.nan
        dx = np.nan
        if self.last_f is not None and self.history:
            df = self.last_f - self.history[-1]["f"]
        if self.last_x is not None and self.history:
            dx = float(np.linalg.norm(self.last_x - self.history[-1]["x"], ord=2))

        rec = dict(iter=self.iter, f=self.last_f, df=df, dx=dx, t=now - self.t0, dt=dt, x=None)
        if self.last_x is not None:
            rec["x"] = self.last_x.copy()
        self.history.append(rec)

        if self.verbose and (self.iter % self.print_every == 0):
            f_str = "nan" if self.last_f is None else f"{self.last_f:.6f}"
            df_str = "nan" if np.isnan(df) else f"{df:+.2e}"
            dx_str = "nan" if np.isnan(dx) else f"{dx:.2e}"
            print(f"[{self.tag}] iter={self.iter}  f={f_str}  Δf={df_str}  |Δx|={dx_str}  {dt:.2f}s")

    def done(self, label="best"):
        if self.verbose:
            if self.history:
                f_best = self.history[-1]["f"]
            else:
                f_best = self.last_f
            total = time.time() - (self.t0 or time.time())
            print(f"[{self.tag} {label}] f={f_best:.6f}  iters={self.iter}  {total:.2f}s")



def _iv_or_nan(S, K, T, r, q, price, cp):
    # Map to no-dividend equivalent for your q-less solver
    S_eff = S * math.exp(-q * T)
    r_eff = r - q

    # No-arbitrage bounds for BS in this transformed setup
    intrinsic = max(0.0, S_eff - K) if cp == "call" else max(0.0, K - S_eff)
    upper = (S_eff if cp == "call" else K * math.exp(-r_eff * T))

    eps = 1e-10
    if not (intrinsic + eps < price < upper - eps):
        return np.nan  # let the objective mask it

    try:
        return _iv_solve(price, S_eff, K, T, r_eff, option_type=cp)
    except Exception:
        return np.nan

def _implied_vol(S, K, T, r, q, price, cp):
    return _iv_or_nan(S, K, T, r, q, price, cp)


# --- vegas for weights (Black–Scholes) ---

# ============================================================
# rBergomi calibration (H, eta, rho, xi0_scalar) per surface
# ============================================================
from src.rough import rbergomi_paths

def _rbergomi_smile_prices_from_paths(ST, r, T, strikes, cp="call"):
    DF = math.exp(-r * T)
    pays = []
    for K in strikes:
        if cp == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        pays.append(float(np.mean(DF * payoff)))
    return np.array(pays, dtype=float)

def _rbergomi_objective(params, data, metric, weights, mc, seed):
    # params = [H, eta, rho, xi0_scalar]  with constraints applied by bounds
    H, eta, rho, xi0 = params
    # penalty to keep xi0 positive and rho in (-1,1)
    if not (-0.999 < rho < 0.999) or xi0 <= 0 or not (0.0 < H < 0.5):
        return 1e6

    err2 = 0.0
    for (S0, r, q, T, strikes, mids, cp) in data:
        # simulate once per maturity, reuse for all K
        t, S_paths, V_paths = rbergomi_paths(
            S0=S0, T=T, N=mc["N"], n_paths=mc["paths"], H=H,
            eta=eta, rho=rho, xi0=xi0, r=r, q=q,
            seed=seed + int(1000 * T), fgn_method=mc.get("fgn_method","davies-harte")
        )
        ST = S_paths[:, -1]
        model_px = _rbergomi_smile_prices_from_paths(ST, r, T, strikes, cp=cp)

        if metric == "price":
            resid = model_px - mids
            if weights is not None and T in weights:
                w = weights[T]
                resid = resid * w
            err2 += float(resid @ resid)
        else:
            # IV space
            mod_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, model_px)])
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
            resid = mod_iv - mkt_iv
            if weights is not None and T in weights:
                w = weights[T]
                resid = resid * w
            err2 += float(resid @ resid)

    return err2

def calibrate_rbergomi(
    smiles,
    metric="iv",
    vega_weight=True,
    bounds=((0.02, 0.45), (0.2, 3.0), (-0.999, -0.01), (1e-4, 1.0)),  # H, eta, rho, xi0
    x0=(0.10, 1.5, -0.7, 0.04),
    mc=dict(N=192, paths=12000, fgn_method="davies-harte"),
    seed=1234,
    options=None,
    multistart=3,
    verbose=True,
    print_every=1,
):
    """
    Calibrate rBergomi (H, eta, rho, xi0 constant) to one or more smiles.

    smiles: list of (S0, r, q, T, strikes, mids, cp) for each maturity
    metric: "iv" or "price"
    vega_weight: if True and metric="iv", multiply residuals by market vegas
                 if metric="price", weights default to identity
    bounds: ((H_lo,H_hi),(eta_lo,eta_hi),(rho_lo,rho_hi),(xi0_lo,xi0_hi))
    x0: initial guess
    mc: dict with N, paths, fgn_method
    seed: base seed for common random numbers
    options: scipy minimize options
    multistart: number of random starts around x0
    """

    # weights per maturity
    weights = None
    if vega_weight and metric == "iv":
        weights = {}
        for (S0, r, q, T, strikes, mids, cp) in smiles:
            # compute market vegas at each strike using market IV from mids
            ivs = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
            vega = np.array([bs_vega(S0, K, T, r, sig, q=q) for K, sig in zip(strikes, ivs)])
            # normalize weights to avoid scale issues
            w = vega / max(1e-8, np.percentile(vega, 75))
            weights[T] = w

    dat = smiles
    b = Bounds([bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0]],
               [bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1]])

    best = None
    rng = np.random.default_rng(999)
    starts = [np.array(x0, dtype=float)]
    for _ in range(max(0, multistart - 1)):
        noise = np.array([0.05, 0.15, 0.05, 0.10]) * (rng.random(4) - 0.5)
        starts.append(np.clip(starts[0] * (1.0 + noise), b.lb, b.ub))

    best_hist = None
    for i, guess in enumerate(starts, 1):
        tag = f"rBergomi #{i}"
        mon = _CalibMonitor(tag=tag, print_every=print_every, verbose=verbose)
        obj = lambda x: _rbergomi_objective(x, dat, metric, weights, mc, seed)
        obj_wrapped = mon.wrap_obj(obj)
        mon.start(start_idx=i)
        res = minimize(
            obj_wrapped,
            x0=np.array(guess),
            method="L-BFGS-B",
            bounds=b,
            options=options or {"maxiter": 200, "disp": False},
            callback=mon.cb,
        )
        mon.done(label="best")
        res.history = mon.history  # attach iteration trace

        if (best is None) or (res.fun < best.fun):
            best = res
            best_hist = mon.history

    out = dict(H=best.x[0], eta=best.x[1], rho=best.x[2], xi0=best.x[3],
               obj=best.fun, success=best.success, nit=best.nit, history=best_hist)
    return out, best


# ============================================================
# Rough Heston calibration (v0, kappa, theta, eta, rho, H)
# ============================================================
from src.rough import rough_heston_euro_mc, rough_heston_paths

def _rough_heston_smile_prices_from_paths(ST, r, T, strikes, cp="call"):
    DF = math.exp(-r * T)
    pays = []
    for K in strikes:
        if cp == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        pays.append(float(np.mean(DF * payoff)))
    return np.array(pays, dtype=float)

def _rough_heston_objective(params, data, metric, weights, mc, seed):
    # params = [v0, kappa, theta, eta, rho, H]
    v0, kappa, theta, eta, rho, H = params
    if v0 <= 0 or theta <= 0 or eta <= 0 or not (-0.999 < rho < 0.999) or not (0.02 < H < 0.5) or kappa <= 0:
        return 1e6

    err2 = 0.0
    for (S0, r, q, T, strikes, mids, cp) in data:
        # simulate once per maturity, reuse for all K
        t, S, V = rough_heston_paths(
            S0=S0, v0=v0, T=T, N=mc["N"], n_paths=mc["paths"],
            H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
            r=r, q=q, seed=seed + int(1000 * T), batch_size=mc.get("batch_size", 1024)
        )
        ST = S[:, -1]
        model_px = _rough_heston_smile_prices_from_paths(ST, r, T, strikes, cp=cp)

        if metric == "price":
            resid = model_px - mids
            if weights is not None and T in weights:
                resid = resid * weights[T]
            err2 += float(resid @ resid)
        else:
            mod_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, model_px)])
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
            resid = mod_iv - mkt_iv
            if weights is not None and T in weights:
                resid = resid * weights[T]
            err2 += float(resid @ resid)

    return err2

def calibrate_rough_heston(
    smiles,
    metric="iv",
    vega_weight=True,
    bounds=((1e-4, 0.5), (0.05, 6.0), (1e-4, 0.5), (0.05, 3.0), (-0.999, -0.01), (0.02, 0.45)),
    x0=(0.04, 1.5, 0.04, 1.8, -0.7, 0.1),
    mc=dict(N=192, paths=12000, batch_size=1024),
    seed=7777,
    options=None,
    multistart=3,
    verbose=True,
    print_every=1,
):
    """
    Calibrate rough Heston (v0, kappa, theta, eta, rho, H) to one or more smiles.

    smiles: list of (S0, r, q, T, strikes, mids, cp) for each maturity
    metric: "iv" or "price"
    vega_weight: if True and metric="iv", multiplies residuals by market vegas
    bounds: parameter bounds tuple
    x0: initial guess
    mc: dict with N, paths, batch_size
    seed: base seed for common random numbers
    options: scipy minimize options
    multistart: number of random starts around x0
    """

    weights = None
    if vega_weight and metric == "iv":
        weights = {}
        for (S0, r, q, T, strikes, mids, cp) in smiles:
            ivs = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
            vega = np.array([bs_vega(S0, K, T, r, sig, q=q) for K, sig in zip(strikes, ivs)])
            w = vega / max(1e-8, np.percentile(vega, 75))
            weights[T] = w

    dat = smiles
    b = Bounds([b[0] for b in bounds], [b[1] for b in bounds])

    best = None
    rng = np.random.default_rng(2024)
    starts = [np.array(x0, dtype=float)]
    for _ in range(max(0, multistart - 1)):
        noise = np.array([0.2, 0.2, 0.2, 0.2, 0.05, 0.1]) * (rng.random(6) - 0.5)
        starts.append(np.clip(starts[0] * (1.0 + noise), b.lb, b.ub))

    best_hist = None
    for i, guess in enumerate(starts, 1):
        tag = f"RoughHeston #{i}"
        mon = _CalibMonitor(tag=tag, print_every=print_every, verbose=verbose)
        obj = lambda x: _rough_heston_objective(x, dat, metric, weights, mc, seed)
        obj_wrapped = mon.wrap_obj(obj)
        mon.start(start_idx=i)
        res = minimize(
            obj_wrapped,
            x0=np.array(guess),
            method="L-BFGS-B",
            bounds=b,
            options=options or {"maxiter": 200, "disp": False},
            callback=mon.cb,
        )
        mon.done(label="best")
        res.history = mon.history

        if (best is None) or (res.fun < best.fun):
            best = res
            best_hist = mon.history

    p = best.x
    out = dict(v0=p[0], kappa=p[1], theta=p[2], eta=p[3], rho=p[4], H=p[5],
               obj=best.fun, success=best.success, nit=best.nit, history=best_hist)
    return out, best
