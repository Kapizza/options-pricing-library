# options_pricing/binomial_tree.py

import numpy as np

def binomial_tree(S, K, T, r, sigma, steps=100, option_type="call", american=False):
    """
    Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree model.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    steps : int
        Number of steps in the binomial tree
    option_type : str
        "call" or "put"
    american : bool
        If True, allows early exercise (American option)

    Returns
    -------
    float
        Option price
    """

    # time per step
    dt = T / steps
    # up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # stock prices at maturity
    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    # option payoff at maturity
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # backward induction
    discount = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        values = discount * (p * values[1:i+2] + (1 - p) * values[0:i+1])
        if american:
            ST = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
            if option_type == "call":
                values = np.maximum(values, ST - K)
            else:  # put
                values = np.maximum(values, K - ST)

    return values[0]
