# tests/test_data_loader.py  (offline: no network access needed)
import numpy as np
import pandas as pd
import pytest

from data.data_loader import get_latest_price, compute_annualized_volatility


def _frame(multi):
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    close = [100.0, 101.0, 99.5, 102.0, 103.5]
    if multi:  # yfinance >= 0.2.48 default: MultiIndex (Price, Ticker) columns
        cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Open", "AAPL")], names=["Price", "Ticker"])
        return pd.DataFrame(np.c_[close, close], index=idx, columns=cols)
    return pd.DataFrame({"Close": close, "Open": close}, index=idx)


@pytest.mark.parametrize("multi", [False, True])
def test_latest_price_and_vol_with_both_column_layouts(multi):
    df = _frame(multi)
    assert get_latest_price(df) == 103.5
    r = np.log(np.array([101.0, 99.5, 102.0, 103.5]) / np.array([100.0, 101.0, 99.5, 102.0]))
    assert compute_annualized_volatility(df) == pytest.approx(r.std(ddof=1) * np.sqrt(252))


def test_multi_ticker_frame_is_rejected():
    # yfinance keeps (Price, Ticker) columns for several tickers even with
    # multi_level_index=False; silently using the first ticker is wrong.
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("Close", "MSFT")], names=["Price", "Ticker"])
    df = pd.DataFrame(np.c_[[300.0, 305.0, 301.0, 309.0, 310.5], [100.0, 101.0, 99.5, 102.0, 103.5]],
                      index=idx, columns=cols)
    with pytest.raises(ValueError, match="one ticker"):
        get_latest_price(df)
    with pytest.raises(ValueError, match="one ticker"):
        compute_annualized_volatility(df)
