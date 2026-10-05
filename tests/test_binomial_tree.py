# options-pricing-library/tests/test_binomial_tree.py

from src.binomial_tree import binomial_tree
from src.black_scholes import black_scholes_price

def test_binomial_vs_black_scholes_call():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    bs_price = black_scholes_price(S, K, T, r, sigma, option_type="call")
    bt_price = binomial_tree(S, K, T, r, sigma, steps=500, option_type="call", american=False)

    # They should be close
    assert abs(bs_price - bt_price) < 1e-2

def test_binomial_vs_black_scholes_put():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    bs_price = black_scholes_price(S, K, T, r, sigma, option_type="put")
    bt_price = binomial_tree(S, K, T, r, sigma, steps=500, option_type="put", american=False)

    # They should be close
    assert abs(bs_price - bt_price) < 1e-2

def test_american_put_greater_than_european_put():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    european_put = binomial_tree(S, K, T, r, sigma, steps=500, option_type="put", american=False)
    american_put = binomial_tree(S, K, T, r, sigma, steps=500, option_type="put", american=True)

    # American put should never be cheaper than European put
    assert american_put >= european_put


def test_invalid_risk_neutral_probability_raises():
    # T=10, sigma=5%, r=10%, 5 steps: p = (e^{r dt} - d)/(u - d) = 2.05 > 1.
    # The tree used to return 15.2 (BS 63.2) instead of failing.
    import pytest
    with pytest.raises(ValueError):
        binomial_tree(100, 100, 10, 0.10, 0.05, steps=5, option_type="call")
