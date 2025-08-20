import numpy as np
from options_pricing.finite_difference import finite_difference
from options_pricing.black_scholes import black_scholes_price

def test_fd_matches_bs_call_cn():
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    fd = finite_difference(S0, K, T, r, sigma, Smax=500, M=400, N=4000,
                           method="crank-nicolson", option="call")
    bs = black_scholes_price(S0, K, T, r, sigma, option_type="call")
    assert np.isclose(fd, bs, rtol=2e-3)  # within ~0.2%
