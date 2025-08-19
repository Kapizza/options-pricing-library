# monte_carlo.py
import numpy as np

def monte_carlo_option_pricing(S0, K, T, r, sigma, option_type="call", n_simulations=100000, seed=None):
    """
    Monte Carlo option pricing for European options under the Black-Scholes model.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of underlying asset
    option_type : str, optional
        "call" or "put" (default is "call")
    n_simulations : int, optional
        Number of Monte Carlo simulations (default is 100,000)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    float
        Estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    # Simulate terminal stock prices under risk-neutral measure
    Z = np.random.standard_normal(n_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Payoff
    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type.lower() == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount back
    price = np.exp(-r * T) * np.mean(payoff)
    return price


if __name__ == "__main__":
    # Quick demo
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    print("Monte Carlo Call:", monte_carlo_option_pricing(S0, K, T, r, sigma, "call", seed=42))
    print("Monte Carlo Put :", monte_carlo_option_pricing(S0, K, T, r, sigma, "put", seed=42))
