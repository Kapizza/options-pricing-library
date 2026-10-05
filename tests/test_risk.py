import math
import numpy as np
import pytest

from src import risk


@pytest.fixture
def positions():
    # Two simple positions with different strikes/vols/sides
    return [
        {"option": "call", "side": +1, "quantity": 2, "S": 100, "K": 100, "T": 0.5, "r": 0.03, "sigma": 0.20, "multiplier": 100},
        {"option": "put",  "side": -1, "quantity": 1, "S": 100, "K": 95,  "T": 0.5, "r": 0.03, "sigma": 0.22, "multiplier": 100},
    ]


def test_aggregate_greeks_additivity(positions):
    # Aggregate should equal sum of per-position contributions
    agg = risk.aggregate_greeks(positions)
    sums = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0, "value": 0.0}
    for p in positions:
        g = risk.greeks_position(p)
        for k in ("delta", "gamma", "theta", "vega", "rho"):
            sums[k] += g[k]
        sums["value"] += risk.price_position(p)
    for k in sums:
        assert math.isfinite(agg[k])
        assert pytest.approx(sums[k], rel=1e-12, abs=1e-9) == agg[k]


def test_scenario_small_shock_matches_full_reval(positions):
    # For sufficiently small spot shock, Taylor vs full reval should match closely
    dS = 1e-4
    taylor = risk.scenario_pnl_delta_gamma(positions, dS=dS)
    full = risk.scenario_revalue(positions, dS=dS)
    assert pytest.approx(full, rel=1e-4, abs=1e-6) == taylor


def test_pnl_attribution_first_order_consistency(positions):
    # Small changes -> residual should be tiny vs total
    S0, sigma0, r0, T0 = 100.0, 0.20, 0.03, 0.50
    S1, sigma1, r1, T1 = 100.1, 0.205, 0.031, 0.50 - 1.0 / 252.0
    out = risk.pnl_attribution_first_order(positions, S0, sigma0, r0, T0, S1, sigma1, r1, T1)
    assert abs(out["residual"]) < 0.05 * max(1.0, abs(out["total"]))


def test_historical_var_es_properties(positions):
    # ES should be >= VaR; both positive (losses) at high alpha; VaR increases with confidence
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0, 0.015, size=3000)
    out_99 = risk.historical_var_es(rets, positions, alpha=0.99)
    out_95 = risk.historical_var_es(rets, positions, alpha=0.95)
    assert out_99["ES"] >= out_99["VaR"] >= 0.0
    assert out_99["VaR"] >= out_95["VaR"]
    assert "pnl_samples" in out_99 and isinstance(out_99["pnl_samples"], np.ndarray)


def test_mc_var_es_delta_gamma_vs_full_reval_close(positions):
    # Under small daily vol, delta-gamma and full-reval VaR should be reasonably close
    params = dict(n_sims=20000, mu=0.0, sigma_ret=0.01, alpha=0.99, seed=7)
    dg = risk.mc_var_es(positions, method="delta_gamma", **params)
    fr = risk.mc_var_es(positions, method="full_reval", **params)
    denom = max(1e-8, fr["VaR"])
    assert abs(dg["VaR"] - fr["VaR"]) / denom < 0.25  # allow some slack
    assert dg["ES"] >= dg["VaR"]
    assert fr["ES"] >= fr["VaR"]


def test_stress_grid_contains_base_and_shapes(positions):
    df = risk.stress_grid(
        positions,
        S_moves=(-0.1, 0.0, 0.1),
        vol_moves=(-0.1, 0.0, 0.1),
        r_moves=(-0.005, 0.0, 0.005),
        horizon_days=1,
    )
    # shape = 3 * 3 * 3 rows
    assert df.shape[0] == 27
    # there should be a row with all zeros (base) with ~0 PnL
    base = df[(df["S_move"] == 0.0) & (df["vol_move"] == 0.0) & (df["r_move"] == 0.0)]
    assert len(base) == 1
    assert abs(float(base["PnL"].iloc[0])) < 1e-8


def test_pnl_attribution_respects_each_legs_maturity():
    # Calendar spread: the two legs have different T. Attribution must move
    # each leg from its own state, so the total equals the full-revaluation P&L.
    cal = [
        {"option": "call", "side": +1, "quantity": 1, "S": 100.0, "K": 100.0, "T": 1.00, "r": 0.03, "sigma": 0.20, "multiplier": 100},
        {"option": "call", "side": -1, "quantity": 1, "S": 100.0, "K": 100.0, "T": 0.25, "r": 0.03, "sigma": 0.20, "multiplier": 100},
    ]
    d = 1.0 / 252.0
    S0, s0, r0, T0 = 100.0, 0.20, 0.03, 0.25
    S1, s1, r1, T1 = 101.0, 0.21, 0.031, 0.25 - d
    out = risk.pnl_attribution_first_order(cal, S0, s0, r0, T0, S1, s1, r1, T1)
    before = sum(risk.price_position(p) for p in cal)
    after = sum(risk.price_position(dict(p, S=p["S"] + 1.0, sigma=p["sigma"] + 0.01,
                                         r=p["r"] + 0.001, T=p["T"] - d)) for p in cal)
    assert abs(out["total"] - (after - before)) < 1e-8
    assert abs(out["delta"] + out["vega"] + out["rho"] + out["theta"] + out["residual"] - out["total"]) < 1e-8


def test_taylor_and_full_reval_agree_on_time_decay(positions):
    # dT is the change in time-to-maturity in both functions (-1/252: a day passes)
    dT = -1.0 / 252.0
    taylor = risk.scenario_pnl_delta_gamma(positions, dT=dT)
    full = risk.scenario_revalue(positions, dT=dT)
    assert full < 0.0
    assert taylor == pytest.approx(full, rel=0.02)


def test_mc_var_methods_use_the_same_horizon(positions):
    # With (almost) no spot moves both methods should report the one-day carry.
    params = dict(n_sims=2000, mu=0.0, sigma_ret=1e-9, alpha=0.99, seed=3)
    dg = risk.mc_var_es(positions, method="delta_gamma", **params)
    fr = risk.mc_var_es(positions, method="full_reval", **params)
    assert np.mean(dg["pnl_samples"]) == pytest.approx(np.mean(fr["pnl_samples"]), rel=0.02)


def test_positions_use_their_dividend_yield():
    from src.black_scholes import black_scholes_price
    from src.greeks import delta, vanna_volga
    pos = {"option": "call", "side": +1, "quantity": 2, "S": 100.0, "K": 100.0, "T": 0.5,
           "r": 0.03, "sigma": 0.2, "q": 0.05, "multiplier": 100}
    m = 200.0
    assert risk.price_position(pos) == pytest.approx(m * black_scholes_price(100, 100, 0.5, 0.03, 0.2, "call", q=0.05))
    assert risk.greeks_position(pos)["delta"] == pytest.approx(m * float(delta(100, 100, 0.5, 0.03, 0.2, "call", q=0.05)))
    hv = vanna_volga(100, 100, 0.5, 0.03, 0.2, q=0.05)
    assert risk.higher_greeks_position(pos)["vanna"] == pytest.approx(m * float(hv["vanna"]))
