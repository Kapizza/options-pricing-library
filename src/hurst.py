# --------------------------------------------------------------------------------------
# Hurst exponent estimators:
#   - hurst_rs:  Rescaled-Range (R/S) method
#   - hurst_dfa: Detrended Fluctuation Analysis (DFA, polynomial order k)
#
# Both return a single float H (np.nan if insufficient data).
# --------------------------------------------------------------------------------------

import numpy as np


def _nanless_1d(x):
    """Return a 1D float array with NaNs removed."""
    x = np.asarray(x, dtype=float).ravel()
    return x[~np.isnan(x)]


def _logspace_int(low, high, num=20):
    """Monotone increasing unique integer grid between low and high (inclusive)."""
    if high < low:
        return np.array([], dtype=int)
    grid = np.unique(np.clip(np.round(np.logspace(np.log10(low), np.log10(high), num=num)), low, high).astype(int))
    return grid


def hurst_rs(series, min_window=8, max_window=None):
    """
    Estimate Hurst exponent H via the classic rescaled range (R/S) method.

    Steps:
      1) For window sizes w in a logarithmic grid, chop the series into ⌊N/w⌋ blocks.
      2) For each block: compute mean-adjusted cumulative sum Z, range R = max(Z)-min(Z),
         and standard deviation S. The block's statistic is (R/S) if S>0.
      3) For each w, average (R/S) over blocks.
      4) Regress log10(E[R/S]_w) on log10(w); slope ≈ H.

    Parameters
    ----------
    series : array-like
        Time series (1D). NaNs are ignored.
    min_window : int, default 8
        Smallest window size.
    max_window : int or None
        Largest window size. If None, set to N//2.

    Returns
    -------
    H : float
        Estimated Hurst exponent in (0,1), or np.nan if not enough data.
    """
    x = _nanless_1d(series)
    n = x.size
    if n < max(2 * min_window, 32):
        return np.nan

    if max_window is None:
        max_window = n // 2
    max_window = max(max_window, min_window + 1)

    windows = _logspace_int(min_window, max_window, num=20)
    rs_vals = []
    log_w = []

    for w in windows:
        nblocks = n // w
        if nblocks < 2:
            continue

        rs_block = []
        for i in range(nblocks):
            seg = x[i * w : (i + 1) * w]
            seg = seg - seg.mean()
            if seg.size < 2:
                continue
            Z = np.cumsum(seg)
            R = Z.max() - Z.min()
            S = seg.std(ddof=1)
            if S > 0:
                rs_block.append(R / S)

        if rs_block:
            rs_vals.append(np.mean(rs_block))
            log_w.append(w)

    if len(rs_vals) < 2:
        return np.nan

    log_w = np.log10(np.asarray(log_w, dtype=float))
    log_rs = np.log10(np.asarray(rs_vals, dtype=float))

    # Simple OLS slope; robust enough for well-behaved grids
    H, _intercept = np.polyfit(log_w, log_rs, 1)
    return float(H)


def hurst_dfa(series, order=1):
    """
    Estimate Hurst exponent H via Detrended Fluctuation Analysis (DFA).

    Canonical DFA steps:
      1) Remove mean from the series and build the profile Y(k) = sum_{i=1..k} (x_i - mean(x)).
      2) For each scale s in a logarithmic grid, split Y into non-overlapping windows of size s
         (drop remainder), and in each window fit a polynomial of given 'order' (default 1=linear).
      3) Compute the variance of (Y - fit) within each window; average across windows to get F(s).
      4) Regress log10(F(s)) on log10(s); slope ≈ H.

    Notes
    -----
    - For uncorrelated noise (Brownian increments / fGn integrated once), DFA returns H ≈ 0.5.
    - For persistent series H>0.5; for anti-persistent H<0.5.
    - Returns np.nan if insufficient data.

    Parameters
    ----------
    series : array-like
        Time series (1D). NaNs are ignored.
    order : int, default 1
        Polynomial detrending order (1=linear, 2=quadratic, ...).

    Returns
    -------
    H : float
        Estimated Hurst exponent, or np.nan if not enough data.
    """
    x = _nanless_1d(series)
    n = x.size
    if n < 32:
        return np.nan

    # Build the "profile"
    y = x - x.mean()
    Y = np.cumsum(y)

    # Choose scales: from 8 up to n//4 (log-spaced)
    s_min = 8
    s_max = max(n // 4, s_min + 1)
    scales = _logspace_int(s_min, s_max, num=20)
    if scales.size < 2:
        return np.nan

    F = []
    S = []

    for s in scales:
        nwin = n // s
        if nwin < 2:
            continue

        # Use only the part that fits an integer number of windows
        Y_cut = Y[: nwin * s].reshape(nwin, s)

        # Detrend each window with a polynomial of given order
        # Fit on local index 0..s-1 to keep conditioning similar across windows
        t = np.arange(s, dtype=float)
        f2_list = []
        for j in range(nwin):
            z = Y_cut[j]
            # If z is constant, variance is zero
            if np.allclose(z, z[0]):
                f2_list.append(0.0)
                continue
            try:
                coeffs = np.polyfit(t, z, deg=order)
                trend = np.polyval(coeffs, t)
                detr = z - trend
                f2 = np.mean(detr * detr)
                f2_list.append(f2)
            except np.linalg.LinAlgError:
                # Fallback: skip this window if ill-conditioned
                continue

        if f2_list:
            F.append(np.sqrt(np.mean(f2_list)))
            S.append(s)

    if len(F) < 2:
        return np.nan

    logS = np.log10(np.asarray(S, dtype=float))
    logF = np.log10(np.asarray(F, dtype=float))

    H, _intercept = np.polyfit(logS, logF, 1)
    return float(H)


# Optional: quick self-test when run directly
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # White noise increments (expect H ~ 0.5)
    x = rng.randn(5000)
    print("R/S H ≈", hurst_rs(x))
    print("DFA H ≈", hurst_dfa(x, order=1))
