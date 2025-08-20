# options-pricing-library/tests/test_binomial_tree.py

from options_pricing.binomial_tree import binomial_tree
from options_pricing.black_scholes import black_scholes_price

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
