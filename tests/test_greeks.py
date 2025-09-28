import numpy as np
from src.greeks import delta, gamma, vega, theta, rho, vanna_volga
from src.black_scholes import black_scholes_price

# -----------------------------
# Baseline: q=0 numeric sanity
# -----------------------------

def test_delta_call_put_q0():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    assert np.isclose(delta(S, K, T, r, sigma, "call"), 0.6368, atol=1e-3)
    assert np.isclose(delta(S, K, T, r, sigma, "put"), -0.3632, atol=1e-3)

def test_vega_q0():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    # vega per 1.00 vol (not per 1%) → ~37.524
    assert np.isclose(vega(S, K, T, r, sigma), 37.524, atol=1e-2)

def test_rho_call_put_q0():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    assert np.isclose(rho(S, K, T, r, sigma, "call"), 53.232, atol=1e-2)
    assert np.isclose(rho(S, K, T, r, sigma, "put"), -41.890, atol=1e-2)

def test_gamma_q0():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    assert np.isclose(gamma(S, K, T, r, sigma), 0.018762, atol=1e-5)

def test_theta_call_put_q0():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    assert np.isclose(theta(S, K, T, r, sigma, "call"), -6.414, atol=1e-3)
    assert np.isclose(theta(S, K, T, r, sigma, "put"), -1.658, atol=1e-3)

# -----------------------------------------
# Dividend yield q>0: finite-diff consistency
# -----------------------------------------

def test_greeks_with_q_finite_diffs():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02

    # base price
    V0_call = black_scholes_price(S, K, T, r, sigma, option_type="call", q=q)
    V0_put  = black_scholes_price(S, K, T, r, sigma, option_type="put",  q=q)

    # finite-diff step sizes
    hS = 1e-3 * S          # ~0.1%
    hσ = 1e-4              # 1bp in vol (absolute)
    hr = 1e-4              # 1bp in rates
    hT = 1e-4              # ~0.0001y ~ 0.0365d

    # --- Delta (∂V/∂S)
    Vp = black_scholes_price(S+hS, K, T, r, sigma, "call", q=q)
    Vm = black_scholes_price(S-hS, K, T, r, sigma, "call", q=q)
    delta_fd = (Vp - Vm) / (2*hS)
    assert np.isclose(delta(S, K, T, r, sigma, "call", q=q), delta_fd, rtol=2e-3, atol=2e-4)

    # --- Gamma (∂²V/∂S²)
    Vpp = black_scholes_price(S+hS, K, T, r, sigma, "call", q=q)
    Vmm = black_scholes_price(S-hS, K, T, r, sigma, "call", q=q)
    gamma_fd = (Vpp - 2*V0_call + Vmm) / (hS**2)
    assert np.isclose(gamma(S, K, T, r, sigma, q=q), gamma_fd, rtol=5e-3, atol=5e-6)

    # --- Vega (∂V/∂σ)
    Vp = black_scholes_price(S, K, T, r, sigma+hσ, "call", q=q)
    Vm = black_scholes_price(S, K, T, r, sigma-hσ, "call", q=q)
    vega_fd = (Vp - Vm) / (2*hσ)
    assert np.isclose(vega(S, K, T, r, sigma, q=q), vega_fd, rtol=2e-3, atol=1e-2)

    # --- Rho (∂V/∂r)
    Vp = black_scholes_price(S, K, T, r+hr, sigma, "call", q=q)
    Vm = black_scholes_price(S, K, T, r-hr, sigma, "call", q=q)
    rho_fd = (Vp - Vm) / (2*hr)
    assert np.isclose(rho(S, K, T, r, sigma, "call", q=q), rho_fd, rtol=2e-3, atol=1e-2)

    # --- Theta (∂V/∂T)    # --- Theta: Θ = ∂V/∂t = -∂V/∂T
    Vp = black_scholes_price(S, K, T+hT, r, sigma, "call", q=q)
    Vm = black_scholes_price(S, K, T-hT, r, sigma, "call", q=q)
    dV_dT = (Vp - Vm) / (2*hT)
    theta_fd = -dV_dT
    assert np.isclose(theta(S, K, T, r, sigma, "call", q=q), theta_fd, rtol=2e-3, atol=1e-2)

    # quick put checks for sign consistency (not strict equality here)
    assert delta(S, K, T, r, sigma, "put", q=q) < 0
    assert rho(S, K, T, r, sigma, "put",  q=q) < 0

# -----------------------------
# Vectorization sanity
# -----------------------------

def test_vectorized_inputs():
    S = np.array([90.0, 100.0, 110.0])
    K, T, r, sigma, q = 100.0, 0.5, 0.02, 0.25, 0.01

    d = delta(S, K, T, r, sigma, "call", q=q)
    g = gamma(S, K, T, r, sigma, q=q)
    v = vega (S, K, T, r, sigma, q=q)
    th= theta(S, K, T, r, sigma, "call", q=q)
    rh= rho  (S, K, T, r, sigma, "call", q=q)
    hv= vanna_volga(S, K, T, r, sigma, q=q)

    assert d.shape == S.shape
    assert g.shape == S.shape
    assert v.shape == S.shape
    assert th.shape == S.shape
    assert rh.shape == S.shape
    assert "vanna" in hv and "volga" in hv
    assert np.all(np.isfinite(hv["vanna"]))
    assert np.all(np.isfinite(hv["volga"]))
