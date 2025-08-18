from options_pricing.black_scholes import black_scholes_price

def test_call_put_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    lhs = call - put
    rhs = S - K * (2.71828**(-r * T))
    assert abs(lhs - rhs) < 1e-2