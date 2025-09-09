# backtesting.py — Minimal, no frills
# Compare Black–Scholes model prices to real option quotes.


import numpy as np
import pandas as pd
from src.black_scholes import black_scholes_price


# --- utilities ---------------------------------------------------------------------------

def year_fraction(date, expiry, day_count='ACT/365'):
    """Compute T in years from date→expiry. Supports 'ACT/365' (default) or 'ACT/252'."""
    d = pd.to_datetime(expiry) - pd.to_datetime(date)
    days = float(getattr(d, 'days', np.nan))
    if not np.isfinite(days):
        return np.nan
    if str(day_count).upper().startswith('ACT/252'):
        return max(0.0, days / 252.0)
    return max(0.0, days / 365.0)


def _prepare(options_df, underlying_df, underlying_col='close', day_count='ACT/365', r=0.0):
    """Merge spot S and compute T. Adds fallback r if missing."""
    df = options_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['expiry'] = pd.to_datetime(df['expiry'])

    und = underlying_df[['date', underlying_col]].copy()
    und['date'] = pd.to_datetime(und['date'])
    und = und.rename(columns={underlying_col: 'S'})

    df = df.merge(und, on='date', how='left')
    df['T'] = [year_fraction(d, e, day_count) for d, e in zip(df['date'], df['expiry'])]

    if 'r' not in df.columns:
        df['r'] = r
    return df


def _finish(df):
    """Compute errors + tiny summary and return (df, summary)."""
    df['error'] = df['theo'] - df['mid']
    df['abs_error'] = df['error'].abs()
    df['squared_error'] = df['error'] ** 2
    df['moneyness'] = df['strike'] / df['S']

    clean = df[['abs_error', 'squared_error', 'error']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean):
        mae = float(clean['abs_error'].mean())
        rmse = float(np.sqrt(clean['squared_error'].mean()))
        bias = float(clean['error'].mean())
        n_obs = int(len(clean))
    else:
        mae = np.nan; rmse = np.nan; bias = np.nan; n_obs = 0
    summary = {'n_obs': n_obs, 'mae': mae, 'rmse': rmse, 'bias': bias}

    col_order = ['date','expiry','option_type','strike','S','T','r','mid','sigma','theo','error','abs_error','squared_error','moneyness']
    df = df[[c for c in col_order if c in df.columns]].copy()
    return df.sort_values(['date','expiry','strike']).reset_index(drop=True), summary


# --- Mode 1: constant sigma -------------------------------------------------------------------

def backtest_bs_constant_sigma(options_df, underlying_df, sigma, r=0.0,
                               day_count='ACT/365', underlying_col='close'):
    """Minimal BS backtest using a constant volatility `sigma`."""
    df = _prepare(options_df, underlying_df, underlying_col, day_count, r)

    theo = []
    sig = float(sigma)
    for S, K, T, rr, typ in zip(df['S'], df['strike'], df['T'], df['r'], df['option_type']):
        if np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and S>0 and K>0 and T>0:
            theo.append(black_scholes_price(S, K, T, rr, sig, option_type=str(typ)))
        else:
            theo.append(np.nan)
    df['sigma'] = sig
    df['theo'] = theo

    df, summary = _finish(df)
    summary['sigma_mode'] = 'constant'
    summary['sigma'] = float(sig)
    return df, summary


# --- Mode 2: rolling realized sigma -----------------------------------------------------------

def realized_vol_series(underlying_df, underlying_col='close', window=21, annualization=252):
    """
    Rolling close-to-close realized vol (log returns), annualized.
    Robust to Series/DataFrame intermediate shapes.
    Returns DataFrame with columns: ['date','sigma'].
    """
    px = underlying_df[['date', underlying_col]].copy()
    px['date'] = pd.to_datetime(px['date'])
    s = px.sort_values('date').set_index('date')[underlying_col]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze("columns")
    s = pd.to_numeric(s, errors='coerce')

    rets = np.log(s).diff()
    vol = rets.rolling(window=window).std() * np.sqrt(float(annualization))

    if isinstance(vol, pd.Series):
        out = vol.rename('sigma').reset_index()
    else:
        col = vol.columns[0]
        out = vol[[col]].rename(columns={col: 'sigma'}).reset_index()
    return out


def backtest_bs_realized_sigma(options_df, underlying_df, window=21, r=0.0,
                               day_count='ACT/365', underlying_col='close', annualization=252):
    """BS backtest using trailing realized volatility σ (rolling log‑return std)."""
    df = _prepare(options_df, underlying_df, underlying_col, day_count, r)

    vol_df = realized_vol_series(underlying_df, underlying_col=underlying_col,
                                 window=window, annualization=annualization)
    df = df.merge(vol_df, on='date', how='left')  # adds 'sigma'

    theo = []
    for S, K, T, rr, typ, sig in zip(df['S'], df['strike'], df['T'], df['r'], df['option_type'], df['sigma']):
        if np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and np.isfinite(sig) and S>0 and K>0 and T>0 and sig>0:
            theo.append(black_scholes_price(S, K, T, rr, float(sig), option_type=str(typ)))
        else:
            theo.append(np.nan)
    df['theo'] = theo

    df, summary = _finish(df)
    summary['sigma_mode'] = 'realized'
    summary['window'] = int(window)
    summary['annualization'] = int(annualization)
    return df, summary


__all__ = [
    'year_fraction',
    'backtest_bs_constant_sigma',
    'realized_vol_series',
    'backtest_bs_realized_sigma',
]
