
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.heston import heston_price, heston_charfunc, heston_call_put
from src.black_scholes import black_scholes_price

def test_charfunc_normalization():
    # phi(0) must be 1
    v = heston_charfunc(0.0, T=1.0, r=0.03, kappa=2.0, theta=0.04, sigma=0.2, v0=0.04, rho=-0.5)
    assert abs(v - 1.0) < 1e-12

def test_put_call_parity():
    S0, K, T, r = 100.0, 100.0, 1.0, 0.03
    kappa, theta, sigma, v0, rho = 2.0, 0.04, 0.5, 0.04, -0.7
    c = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="call")
    p = heston_price(S0, K, T, r, kappa, theta, sigma, v0, rho, option="put")
    lhs = c - p
    rhs = S0 - K * np.exp(-r*T)
    assert abs(lhs - rhs) < 1e-6

def test_heston_approaches_bs_in_low_volofvol_limit():
    # When variance is (nearly) constant: kappa large, sigma small, v0 ~= theta, rho ~= 0
    S0, K, T, r = 100.0, 100.0, 1.0, 0.01
    iv = 0.2
    theta = iv**2
    params = dict(kappa=8.0, theta=theta, sigma=1e-3, v0=theta, rho=0.0)
    c_heston = heston_price(S0, K, T, r, **params, option="call", alpha=1.5, N=4096, umax=200.0)
    c_bs     = black_scholes_price(S0, K, T, r, iv, option_type="call")
    assert abs(c_heston - c_bs) < 5e-3  # tight since transform is stable here

def test_strike_monotonicity_calls():
    # Call price decreases with strike (weakly) for fixed params
    S0, T, r = 100.0, 1.0, 0.01
    params = dict(kappa=2.0, theta=0.04, sigma=0.5, v0=0.04, rho=-0.5)
    Ks = [80, 90, 100, 110, 120]
    prices = [heston_price(S0, K, T, r, **params, option="call") for K in Ks]
    assert all(prices[i] >= prices[i+1] - 1e-8 for i in range(len(prices)-1))


# ---------------------------------------------------------------------------
# Regression: accuracy against published and independent reference prices
# ---------------------------------------------------------------------------
import math
import pytest
from scipy.integrate import quad
from src.heston import heston_smile_prices

# Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2), Heston test case
_FO = dict(kappa=1.5768, theta=0.0398, sigma=0.5751, v0=0.0175, rho=-0.5711)


def _heston_gil_pelaez(S0, K, T, r, q, kappa, theta, sigma, v0, rho):
    """Independent Heston call price: Gil-Pelaez inversion of the
    log-price characteristic function (Albrecher et al. 2007 form)."""
    def cf(u):
        iu = 1j * u
        d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma ** 2 * (iu + u * u))
        g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
        e = np.exp(-d * T)
        C = (iu * (math.log(S0) + (r - q) * T)
             + kappa * theta / sigma ** 2 * ((kappa - rho * sigma * iu - d) * T
                                            - 2.0 * np.log((1 - g * e) / (1 - g))))
        D = (kappa - rho * sigma * iu - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0)

    lnK = math.log(K)
    phi_mi = cf(-1j)
    f1 = lambda u: (np.exp(-1j * u * lnK) * cf(u - 1j) / (1j * u * phi_mi)).real
    f2 = lambda u: (np.exp(-1j * u * lnK) * cf(u) / (1j * u)).real
    P1 = 0.5 + quad(f1, 1e-10, 500, limit=2000, epsabs=1e-12)[0] / math.pi
    P2 = 0.5 + quad(f2, 1e-10, 500, limit=2000, epsabs=1e-12)[0] / math.pi
    return S0 * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2


@pytest.mark.parametrize("T, ref", [(1.0, 5.785155450), (10.0, 22.318945791474590)])
def test_fang_oosterlee_reference_values(T, ref):
    assert abs(heston_price(100.0, 100.0, T, 0.0, **_FO) - ref) < 1e-6
    assert abs(heston_smile_prices(100.0, 0.0, 0.0, T, [100.0], **_FO)[0] - ref) < 1e-6


@pytest.mark.parametrize("q", [0.0, 0.02])
def test_strike_sweep_matches_independent_integration(q):
    S0, T, r = 100.0, 0.5, 0.03
    P = dict(kappa=2.0, theta=0.04, sigma=0.6, v0=0.05, rho=-0.7)
    Ks = np.array([50.0, 70.0, 90.0, 100.0, 110.0, 130.0, 160.0])
    ref = np.array([_heston_gil_pelaez(S0, K, T, r, q, **P) for K in Ks])
    smile = heston_smile_prices(S0, r, q, T, Ks, **P)
    assert np.max(np.abs(smile - ref)) < 1e-5
    if q == 0.0:
        single = np.array([heston_price(S0, K, T, r, **P) for K in Ks])
        assert np.max(np.abs(single - ref)) < 1e-5


@pytest.mark.parametrize("sigma", [0.0, 1e-4, 0.01, 0.05])
def test_low_vol_of_vol_with_v0_not_theta(sigma):
    # With v0 != theta the price must reflect the time-averaged variance,
    # not sqrt(theta). Reference: independent integration (sigma >= 0.01)
    # or Black-Scholes with the averaged variance (sigma -> 0 limit).
    S0, K, T, r = 100.0, 100.0, 1.0, 0.0
    kappa, theta, v0 = 2.0, 0.04, 0.09
    avg_var = theta + (v0 - theta) * (1.0 - math.exp(-kappa * T)) / (kappa * T)
    px = heston_price(S0, K, T, r, kappa, theta, sigma, v0, 0.0)
    if sigma < 1e-3:
        ref = black_scholes_price(S0, K, T, r, math.sqrt(avg_var), option_type="call")
        assert abs(px - ref) < 1e-3
    else:
        ref = _heston_gil_pelaez(S0, K, T, r, 0.0, kappa, theta, sigma, v0, 0.0)
        assert abs(px - ref) < 1e-5


# ---------------------------------------------------------------------------
# Regression: kappa = 0 (no mean reversion)
# ---------------------------------------------------------------------------

def test_kappa_zero_matches_small_kappa_and_is_not_black_scholes():
    # At kappa = 0 the closed-form CF is 0/0 at u = 0, so c2 was NaN; the NaN
    # was clamped to 1e-12, which triggered the degenerate fallback and both
    # pricers returned BS(sqrt(v0)) = 7.965567 whatever sigma and rho were.
    # Reference 5.950734939: Lewis (2001) single-integral price at kappa = 0
    # (scratch script; adaptive and Gauss-Legendre quadrature agree to 1e-13,
    # and _heston_gil_pelaez above gives 5.9507349387).
    S0, K, T, r = 100.0, 100.0, 1.0, 0.0
    P = dict(theta=0.04, sigma=0.5, v0=0.04, rho=-0.7)
    ref = 5.950734939
    assert heston_charfunc(0.0, T, r, 0.0, **P) == 1.0
    p0 = heston_price(S0, K, T, r, 0.0, **P)
    p9 = heston_price(S0, K, T, r, 1e-9, **P)
    s0 = heston_smile_prices(S0, r, 0.0, T, [K], kappa=0.0, **P)[0]
    s9 = heston_smile_prices(S0, r, 0.0, T, [K], kappa=1e-9, **P)[0]
    assert abs(p0 - p9) < 1e-6 and abs(s0 - s9) < 1e-6
    assert abs(p0 - ref) < 1e-6 and abs(s0 - ref) < 1e-6
    bs = black_scholes_price(S0, K, T, r, math.sqrt(P["v0"]), option_type="call")
    assert abs(p0 - bs) > 1.0 and abs(s0 - bs) > 1.0
    # strike vector with dividends
    Ks = np.array([80.0, 100.0, 125.0])
    s0 = heston_smile_prices(S0, 0.02, 0.01, T, Ks, kappa=0.0, **P)
    s9 = heston_smile_prices(S0, 0.02, 0.01, T, Ks, kappa=1e-9, **P)
    assert np.max(np.abs(s0 - s9)) < 1e-6
