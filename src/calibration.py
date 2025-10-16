"""
Calibration helpers for rough Heston and rBergomi.

Designed for speed and stability:
- Uses persistent executors across optimizer iterations to avoid pool churn.
- Can return only terminal ST during calibration to reduce IPC.
- Vectorized payoff aggregation across strikes.
- Precomputes market IVs and vega weights once per smile.
"""

import math
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize, Bounds

from src.volatility import implied_volatility as _iv_solve
from src.greeks import vega as bs_vega
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from src.rough import (
    rbergomi_paths,
    rbergomi_paths_parallel,
    rbergomi_paths_parallel_pool,
    rbergomi_terminal_parallel_pool,
    rough_heston_paths_parallel,
    rough_heston_paths_parallel_pool,
    rough_heston_terminal_parallel_pool,
    rough_heston_euro_mc,
)

# -----------------------------
# Generic caching infrastructure
# -----------------------------
import os, json, hashlib, inspect

def _to_ser(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: _to_ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_ser(v) for v in x]
    return x

def _hash_config(d):
    s = json.dumps(_to_ser(d), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:16]

def _cache_file(cache_dir: str, model: str, key: str) -> str:
    d = os.path.join(cache_dir, model)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.json")

def _apply_supported_kwargs(fn, base_kwargs: dict, extra: dict | None):
    """Return kwargs filtered to fn's signature, merging base and extras."""
    out = dict(base_kwargs)
    if extra:
        sig = inspect.signature(fn).parameters
        for k, v in extra.items():
            if k in sig:
                out[k] = v
    return out

def calibrate_cached(
    model: str,
    calibrate_fn,
    *,
    smiles,
    metric,
    vega_weight,
    x0,
    mc,
    seed=None,
    bounds=None,
    multistart=1,
    options=None,
    cache_dir: str | None = None,
    runtime_overrides: dict | None = None,
):
    """
    Generic calibrate-with-cache helper.

    - model: 'rbergomi' or 'rough_heston' (used for cache subfolder)
    - calibrate_fn: function like calibrate_rbergomi or calibrate_rough_heston
    - runtime_overrides: optional kwargs (e.g., n_workers, parallel_backend, terminal_only)
                         Only supported keys (by signature) are passed.
    - cache_dir: folder to store cache (default './cache')
    """
    if cache_dir is None:
        cache_dir = os.path.abspath(os.path.join(os.getcwd(), "cache"))

    cfg = dict(
        smiles=smiles,
        metric=metric,
        vega_weight=vega_weight,
        x0=x0,
        mc=mc,
        seed=seed,
        bounds=bounds,
        multistart=multistart,
        options=options,
    )
    key = _hash_config({"model": model, **cfg})
    fpath = _cache_file(cache_dir, model, key)
    if os.path.exists(fpath):
        blob = json.load(open(fpath, "r", encoding="utf-8"))
        blob.setdefault("cache_file", fpath)
        print(f"[cache hit] {model} → {os.path.relpath(fpath)}")
        return blob["best"], blob

    t0 = time.time()
    base_kwargs = dict(metric=metric, vega_weight=vega_weight, x0=x0, mc=mc, seed=seed, bounds=bounds, multistart=multistart, options=options)
    kwargs = _apply_supported_kwargs(calibrate_fn, base_kwargs, runtime_overrides)
    best, raw = calibrate_fn(smiles, **kwargs)
    dt = time.time() - t0
    blob = {"best": best, "raw": {}, "elapsed_sec": dt, "cfg": _to_ser(cfg), "cache_file": fpath}
    json.dump(_to_ser(blob), open(fpath, "w", encoding="utf-8"), indent=2)
    print(f"[cache saved] {model} ({dt:.2f}s) → {os.path.relpath(fpath)}")
    return best, blob


class _CalibMonitor:
    def __init__(self, tag: str = "calib", print_every: int = 1, verbose: bool = True):
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
            print(f"[{self.tag}] iter={self.iter}  f={f_str}  df={df_str}  |dx|={dx_str}  {dt:.2f}s")

    def done(self, label="best"):
        if self.verbose:
            if self.history:
                f_best = self.history[-1]["f"]
            else:
                f_best = self.last_f
            total = time.time() - (self.t0 or time.time())
            print(f"[{self.tag} {label}] f={f_best:.6f}  iters={self.iter}  {total:.2f}s")


def _iv_or_nan(S, K, T, r, q, price, cp):
    # Map to no-dividend equivalent for the solver which assumes q=0
    S_eff = S * math.exp(-q * T)
    r_eff = r - q

    intrinsic = max(0.0, S_eff - K) if cp == "call" else max(0.0, K - S_eff)
    upper = (S_eff if cp == "call" else K * math.exp(-r_eff * T))

    eps = 1e-10
    if not (intrinsic + eps < price < upper - eps):
        return np.nan
    try:
        return _iv_solve(price, S_eff, K, T, r_eff, option_type=cp)
    except Exception:
        return np.nan


def _implied_vol(S, K, T, r, q, price, cp):
    return _iv_or_nan(S, K, T, r, q, price, cp)


def _smile_from_ST(ST, r, T, strikes, cp="call"):
    DF = math.exp(-r * T)
    ST = np.asarray(ST, dtype=float).reshape(-1)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    if cp == "call":
        payoff = np.maximum(ST[:, None] - K[None, :], 0.0)
    else:
        payoff = np.maximum(K[None, :] - ST[:, None], 0.0)
    return DF * payoff.mean(axis=0)


# ============================= rBergomi =============================

def _rbergomi_objective(params, data, metric, weights, mc, seed, exec_ctx=None, terminal_only=True):
    # params = [H, eta, rho, xi0]
    H, eta, rho, xi0 = params
    if not (-0.999 < rho < 0.999) or xi0 <= 0 or not (0.0 < H < 0.5):
        return 1e6

    err2 = 0.0
    for (S0, r, q, T, strikes, mids, cp, mkt_iv_opt) in data:
        base_seed = seed + int(1000 * T)
        if exec_ctx is None:
            t, S_paths, _V = rbergomi_paths_parallel(
                S0=S0, T=T, N=mc["N"], n_paths=mc["paths"], H=H, eta=eta, rho=rho, xi0=xi0,
                r=r, q=q, base_seed=base_seed, n_workers=mc.get("n_workers", 4),
                fgn_method=mc.get("fgn_method", "davies-harte")
            )
            ST = S_paths[:, -1]
        else:
            if terminal_only:
                ST = rbergomi_terminal_parallel_pool(
                    exec_ctx,
                    S0=S0, T=T, N=mc["N"], n_paths=mc["paths"], H=H, eta=eta, rho=rho, xi0=xi0,
                    r=r, q=q, base_seed=base_seed, fgn_method=mc.get("fgn_method", "davies-harte"),
                    batch_size=mc.get("batch_size", 8192)
                )
            else:
                _t, S_paths, _V = rbergomi_paths_parallel_pool(
                    exec_ctx,
                    S0=S0, T=T, N=mc["N"], n_paths=mc["paths"], H=H, eta=eta, rho=rho, xi0=xi0,
                    r=r, q=q, base_seed=base_seed, fgn_method=mc.get("fgn_method", "davies-harte"),
                    batch_size=mc.get("batch_size", 8192), return_variance=False
                )
                ST = S_paths[:, -1]

        model_px = _smile_from_ST(ST, r, T, strikes, cp=cp)

        if metric == "price":
            resid = model_px - mids
            if weights is not None and T in weights:
                resid = resid * weights[T]
            err2 += float(resid @ resid)
        else:
            mod_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, model_px)])
            mkt_iv = mkt_iv_opt
            resid = mod_iv - mkt_iv
            if weights is not None and T in weights:
                resid = resid * weights[T]
            err2 += float(resid @ resid)

    return err2


def calibrate_rbergomi(
    smiles: List[Tuple],
    metric: str = "iv",
    vega_weight: bool = True,
    bounds=((0.02, 0.45), (0.2, 3.0), (-0.999, -0.01), (1e-4, 1.0)),  # H, eta, rho, xi0
    x0=(0.10, 1.5, -0.7, 0.04),
    mc=dict(N=192, paths=12000, fgn_method="davies-harte", batch_size=8192, n_workers=4),
    seed: int = 1234,
    n_workers: int = 4,
    parallel_backend: str = "process",  # or "thread"
    terminal_only: bool = True,
    options=None,
    multistart: int = 3,
    verbose: bool = True,
    print_every: int = 1,
):
    """
    Calibrate rBergomi (H, eta, rho, xi0 constant) to one or more smiles.
    smiles: list of (S0, r, q, T, strikes, mids, cp)
    """
    # Allow overriding workers via explicit param for backward compat
    if n_workers is not None:
        mc = dict(mc)
        mc["n_workers"] = int(n_workers)

    # Precompute market IVs and vega weights
    dat = []
    for (S0, r, q, T, strikes, mids, cp) in smiles:
        if metric == "iv":
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
        else:
            mkt_iv = None
        dat.append((S0, r, q, T, np.asarray(strikes, float), np.asarray(mids, float), cp, mkt_iv))

    weights = None
    if vega_weight and metric == "iv":
        weights = {}
        for (S0, r, q, T, strikes, mids, cp, mkt_iv) in dat:
            ivs = mkt_iv
            vega = np.array([bs_vega(S0, K, T, r, sig, q=q) for K, sig in zip(strikes, ivs)])
            w = vega / max(1e-8, np.percentile(vega, 75))
            weights[T] = w

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

        max_workers = int(mc.get("n_workers", 4))
        Executor = ThreadPoolExecutor if str(parallel_backend).lower().startswith("thread") else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            obj = lambda x: _rbergomi_objective(x, dat, metric, weights, mc, seed, ex, terminal_only)
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

    out = dict(H=best.x[0], eta=best.x[1], rho=best.x[2], xi0=best.x[3],
               obj=best.fun, success=best.success, nit=best.nit, history=best_hist)
    return out, best


# =========================== Rough Heston ===========================

def _rough_heston_objective(params, data, metric, weights, mc, seed, exec_ctx=None, terminal_only=True):
    # params = [v0, kappa, theta, eta, rho, H]
    v0, kappa, theta, eta, rho, H = params
    if v0 <= 0 or theta <= 0 or eta <= 0 or not (-0.999 < rho < 0.999) or not (0.02 < H < 0.5) or kappa <= 0:
        return 1e6

    err2 = 0.0
    for (S0, r, q, T, strikes, mids, cp, mkt_iv_opt) in data:
        base_seed = seed + int(1000 * T)
        use_numba_flag = mc.get("use_numba", None)
        if exec_ctx is None:
            t, S, V = rough_heston_paths_parallel(
                S0=S0, v0=v0, T=T, N=mc["N"], n_paths=mc["paths"], H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
                r=r, q=q, base_seed=base_seed, n_workers=mc.get("n_workers", 4),
                batch_size=mc.get("batch_size", 1024), use_numba=use_numba_flag
            )
            ST = S[:, -1]
        else:
            if terminal_only:
                ST = rough_heston_terminal_parallel_pool(
                    exec_ctx,
                    S0=S0, v0=v0, T=T, N=mc["N"], n_paths=mc["paths"], H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
                    r=r, q=q, base_seed=base_seed, batch_size=mc.get("batch_size", 1024),
                    use_numba=use_numba_flag
                )
            else:
                _t, S, _V = rough_heston_paths_parallel_pool(
                    exec_ctx,
                    S0=S0, v0=v0, T=T, N=mc["N"], n_paths=mc["paths"], H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
                    r=r, q=q, base_seed=base_seed, batch_size=mc.get("batch_size", 1024),
                    use_numba=use_numba_flag
                )
                ST = S[:, -1]

        model_px = _smile_from_ST(ST, r, T, strikes, cp=cp)

        if metric == "price":
            resid = model_px - mids
            if weights is not None and T in weights:
                resid = resid * weights[T]
            err2 += float(resid @ resid)
        else:
            mod_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, model_px)])
            mkt_iv = mkt_iv_opt
            resid = mod_iv - mkt_iv
            if weights is not None and T in weights:
                resid = resid * weights[T]
            err2 += float(resid @ resid)

    return err2


def calibrate_rough_heston(
    smiles: List[Tuple],
    metric: str = "iv",
    vega_weight: bool = True,
    bounds=((1e-4, 0.5), (0.05, 6.0), (1e-4, 0.5), (0.05, 3.0), (-0.999, -0.01), (0.02, 0.45)),
    x0=(0.04, 1.5, 0.04, 1.8, -0.7, 0.1),
    mc=dict(N=192, paths=12000, batch_size=1024, n_workers=4, use_numba=True),
    seed: int = 7777,
    n_workers: int = 4,
    parallel_backend: str = "process",
    terminal_only: bool = True,
    options=None,
    multistart: int = 3,
    verbose: bool = True,
    print_every: int = 1,
):
    """
    Calibrate rough Heston (v0, kappa, theta, eta, rho, H) to one or more smiles.
    smiles: list of (S0, r, q, T, strikes, mids, cp)
    """
    if n_workers is not None:
        mc = dict(mc)
        mc["n_workers"] = int(n_workers)

    dat = []
    for (S0, r, q, T, strikes, mids, cp) in smiles:
        if metric == "iv":
            mkt_iv = np.array([_implied_vol(S0, K, T, r, q, p, cp) for K, p in zip(strikes, mids)])
        else:
            mkt_iv = None
        dat.append((S0, r, q, T, np.asarray(strikes, float), np.asarray(mids, float), cp, mkt_iv))

    weights = None
    if vega_weight and metric == "iv":
        weights = {}
        for (S0, r, q, T, strikes, mids, cp, mkt_iv) in dat:
            ivs = mkt_iv
            vega = np.array([bs_vega(S0, K, T, r, sig, q=q) for K, sig in zip(strikes, ivs)])
            w = vega / max(1e-8, np.percentile(vega, 75))
            weights[T] = w

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

        max_workers = int(mc.get("n_workers", 4))
        Executor = ThreadPoolExecutor if str(parallel_backend).lower().startswith("thread") else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as ex:
            obj = lambda x: _rough_heston_objective(x, dat, metric, weights, mc, seed, ex, terminal_only)
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
