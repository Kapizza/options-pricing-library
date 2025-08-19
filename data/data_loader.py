import yfinance as yf
import numpy as np
import pandas as pd

def fetch_stock_data(ticker, start="2020-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    return df
def compute_annualized_volatility(stock_data):
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    sigma = log_returns.dropna().std(axis=0) * np.sqrt(252)
    return sigma.item()  # safe scalar

def get_latest_price(stock_data):
    return float(stock_data['Close'].values[-1])

