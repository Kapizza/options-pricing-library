# options_pricing/util.py

import numpy as np
import pandas as pd


__all__ = [
    "month_first_trading_days",
    "make_roll_calendar",
    "rolling_annualized_vol",
    "remaining_T_in_years",
    "infer_atm_strike",
    "align_series_to_index",
]


# -----------------------------
# Calendar / roll-date helpers
# -----------------------------
def month_first_trading_days(index):
    """
    Return the first trading day of each month from a DatetimeIndex of trading days.
    Uses a duplicated-mask instead of Index.groupby to avoid PrettyDict issues.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    # One Period per row; first occurrence in each month is where duplicated is False
    periods = index.to_period("M")
    mask = ~periods.duplicated()
    firsts = index[mask]
    return pd.DatetimeIndex(firsts)


def make_roll_calendar(index, schedule="monthly", weekday=0):
    """
    Build a roll calendar from a trading-day DatetimeIndex.

    schedule:
      - "monthly": first trading day each month
      - "weekly": all trading days that fall on the given weekday (0=Mon,...,4=Fri)
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    if schedule == "monthly":
        return month_first_trading_days(index)

    if schedule == "weekly":
        weekday = int(weekday)
        if not (0 <= weekday <= 6):
            raise ValueError("weekday must be in [0..6]; 0=Mon, 6=Sun")
        # If your index only contains trading days, this keeps the chosen weekday trading sessions
        mask = index.weekday == weekday
        return pd.DatetimeIndex(index[mask])

    raise ValueError("schedule must be 'monthly' or 'weekly'")

# -----------------------------
# Volatility helpers
# -----------------------------
def rolling_annualized_vol(close, window=21, min_periods=None, annualization=252, method="log"):
    """
    Rolling realized volatility:
      - log returns (default) or simple returns
      - annualized by sqrt(annualization)

    Parameters
    ----------
    close : pd.Series
        Close price series indexed by trading days.
    window : int
        Rolling window in trading days (e.g., 21 ~ 1 month).
    min_periods : int or None
        Minimum observations required for a value; defaults to 'window'.
    annualization : int
        Trading days in a year (252 by default).
    method : str
        "log" for log returns; "simple" for arithmetic returns.

    Returns
    -------
    pd.Series
        Annualized rolling volatility (σ) aligned to 'close.index'.
    """
    if min_periods is None:
        min_periods = window

    if method == "log":
        rets = np.log(close).diff()
    elif method == "simple":
        rets = close.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    vol = rets.rolling(window=window, min_periods=min_periods).std() * np.sqrt(float(annualization))
    # Back/forward-fill edges so downstream code doesn't crash on NaNs
    return vol.bfill().ffill()


# -----------------------------
# Time-to-maturity helpers
# -----------------------------
def remaining_T_in_years(start_date, expiry_date, current_date, basis=252):
    """
    Remaining time to maturity in years using a simple day-count basis.

    Notes:
      - Uses calendar days between dates divided by 'basis' (default 252).
      - Keep it simple; if you prefer strict trading-day counts, plug one in.
    """
    days_rem = (pd.Timestamp(expiry_date) - pd.Timestamp(current_date)).days
    return max(0.0, days_rem / float(basis))


# -----------------------------
# Strike / alignment helpers
# -----------------------------
def infer_atm_strike(S, step=None):
    """
    Infer an ATM strike from spot S.
      - If step is None: return float(S)
      - If step is provided (e.g., 1, 2.5, 5): round to nearest multiple
    """
    S = float(S)
    if step is None or step <= 0:
        return S
    return round(S / step) * float(step)


def align_series_to_index(series, index, method="ffill"):
    """
    Align a (possibly sparse) Series to a target index using a fill method.

    Parameters
    ----------
    series : pd.Series
    index  : pd.DatetimeIndex
    method : "ffill" or "bfill"

    Returns
    -------
    pd.Series
        Series reindexed to 'index' with fills applied.
    """
    s = pd.Series(series).copy()
    s = s.reindex(pd.DatetimeIndex(index))
    if method == "ffill":
        return s.ffill().bfill()
    if method == "bfill":
        return s.bfill().ffill()
    raise ValueError("method must be 'ffill' or 'bfill'")
