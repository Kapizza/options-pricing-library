# tests/test_utils.py
import pandas as pd
import pytest

from src.utils import remaining_T_in_years
from src.backtesting import year_fraction


def test_remaining_T_trading_day_basis_counts_business_days():
    # One calendar year must be ~1 year on a 252 trading-day basis (weekdays,
    # no holiday calendar: 262/252), not 366/252 = 1.45.
    T = remaining_T_in_years("2024-01-01", "2025-01-01", "2024-01-01", basis=252)
    assert abs(T - 1.0) < 0.05
    # one trading week
    assert remaining_T_in_years("2024-01-01", "2024-01-08", "2024-01-01") == pytest.approx(5 / 252)
    # calendar basis unchanged
    assert remaining_T_in_years("2024-01-01", "2025-01-01", "2024-01-01", basis=365) == pytest.approx(366 / 365)
    assert remaining_T_in_years("2024-01-01", "2023-12-01", "2024-01-01") == 0.0


def test_year_fraction_act252_counts_business_days():
    assert abs(year_fraction("2024-01-01", "2025-01-01", "ACT/252") - 1.0) < 0.05
    assert year_fraction("2024-01-01", "2025-01-01") == pytest.approx(366 / 365)
