import yfinance as yf
import numpy as np
import pandas as pd

def fetch_stock_data(ticker, start="2020-01-01", end=None):
    # flat columns ("Close", ...) rather than yfinance's default (Price, Ticker) MultiIndex
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, multi_level_index=False)
    df.index = pd.to_datetime(df.index)
    return df

def _close_series(stock_data):
    """'Close' column as a Series, for flat or (Price, Ticker) MultiIndex columns."""
    close = stock_data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close

def compute_annualized_volatility(stock_data):
    close = _close_series(stock_data)
    log_returns = np.log(close / close.shift(1))
    sigma = log_returns.dropna().std() * np.sqrt(252)
    return float(sigma)

def get_latest_price(stock_data):
    return float(_close_series(stock_data).iloc[-1])

def fetch_current_chain(ticker):
    yf_t = yf.Ticker(ticker)
    expiries = yf_t.options
    rows = []
    for ex in expiries:
        chain = yf_t.option_chain(ex)
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty: 
                continue
            for _, row in df.iterrows():
                # mid price (skip junk)
                if pd.isna(row.get("lastPrice")): 
                    continue
                bid = row.get("bid", np.nan); ask = row.get("ask", np.nan)
                mid = np.nanmean([bid, ask]) if np.isfinite(bid) and np.isfinite(ask) else row["lastPrice"]
                rows.append({"expiration": pd.to_datetime(ex), "strike": float(row["strike"]), "type": side, "mid": float(mid)})
    return pd.DataFrame(rows).dropna()