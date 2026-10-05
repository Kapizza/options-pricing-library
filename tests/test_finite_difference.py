import numpy as np
from src.finite_difference import finite_difference
from src.black_scholes import black_scholes_price

def test_fd_matches_bs_call_cn():
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    fd = finite_difference(S0, K, T, r, sigma, Smax=500, M=400, N=4000,
                           method="crank-nicolson", option="call")
    bs = black_scholes_price(S0, K, T, r, sigma, option_type="call")
    assert np.isclose(fd, bs, rtol=2e-3)  # within ~0.2%


# --- Regression: grid sizing and explicit-scheme stability ---
import pytest


@pytest.mark.parametrize("S0, K, T, sigma", [(250.0, 250.0, 1.0, 0.2), (1000.0, 900.0, 0.5, 0.3), (100.0, 100.0, 2.0, 0.4)])
def test_default_grid_scales_with_spot_and_strike(S0, K, T, sigma):
    # The old default Smax=200 gave -37.8 for S0=K=250 (BS 26.13).
    for option in ("call", "put"):
        fd = finite_difference(S0, K, T, 0.05, sigma, method="crank-nicolson", option=option)
        bs = black_scholes_price(S0, K, T, 0.05, sigma, option_type=option)
        assert abs(fd - bs) < 2e-3 * max(bs, 1.0)


def test_spot_outside_grid_raises():
    with pytest.raises(ValueError):
        finite_difference(250.0, 100.0, 1.0, 0.05, 0.2, Smax=200, method="crank-nicolson")


@pytest.mark.parametrize("T, sigma", [(2.0, 0.2), (1.0, 0.3)])
def test_explicit_scheme_rejects_unstable_steps(T, sigma):
    # dt = T/2000 violates dt*(2 sigma^2 (M-1)^2 + r) <= 2 here; the scheme used
    # to blow up silently to NaN.
    with pytest.raises(ValueError):
        finite_difference(100.0, 100.0, T, 0.05, sigma, Smax=200, M=200, N=2000, method="explicit")
    fd = finite_difference(100.0, 100.0, T, 0.05, sigma, Smax=200, M=200, N=4500, method="explicit")
    assert np.isfinite(fd)
    assert abs(fd - black_scholes_price(100.0, 100.0, T, 0.05, sigma, option_type="call")) < 0.15
