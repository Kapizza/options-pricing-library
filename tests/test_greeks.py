import numpy as np
from src.greeks import delta, gamma, vega, theta, rho

def test_delta_call_put():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    assert np.isclose(delta(S, K, T, r, sigma, "call"), 0.6368, atol=1e-3)
    assert np.isclose(delta(S, K, T, r, sigma, "put"), -0.3632, atol=1e-3)


def test_vega():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    assert np.isclose(vega(S, K, T, r, sigma), 37.524, atol=1e-2)

def test_rho_call_put():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    assert np.isclose(rho(S, K, T, r, sigma, "call"), 53.232, atol=1e-2)
    assert np.isclose(rho(S, K, T, r, sigma, "put"), -41.890, atol=1e-2)

def test_gamma():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    assert np.isclose(gamma(S, K, T, r, sigma), 0.018762, atol=1e-5)

def test_theta_call_put():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    assert np.isclose(theta(S, K, T, r, sigma, "call"), -6.414, atol=1e-3)
    assert np.isclose(theta(S, K, T, r, sigma, "put"), -1.658, atol=1e-3)
