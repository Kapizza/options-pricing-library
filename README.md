# Options Pricing Library

A Python library for option pricing, Greeks, strategies, volatility modeling, risk management, calibration, and backtesting — with clean implementations of classic models and a full suite of demo notebooks.

This project is designed as an educational and demonstrational toolkit for quantitative finance.

---

## Theory Reference

For background on the Central Limit Theorem (CLT), Brownian Motion, Ito's Lemma, and the Black–Scholes model:

**[`notebooks/00_Theory-CLT-ITO-BS.ipynb`](notebooks/00_Theory-CLT-ITO-BS.ipynb)**


---

## Features

- Core pricing models
  - Black–Scholes closed form (with dividend yield `q`) and implied volatility
  - Binomial Tree (European & American, CRR)
  - Monte Carlo simulation
  - Finite Difference PDE solvers (explicit / implicit / Crank–Nicolson); the grid is sized from
    spot, strike and volatility by default, and the explicit scheme checks its stability bound
  - American options via Longstaff–Schwartz (LSMC)
  - Merton (1976) jump-diffusion via the COS method

- Stochastic volatility models
  - Heston via the COS method (Fang & Oosterlee 2008), checked against their published reference prices
  - SABR: Hagan et al. (2002) lognormal implied vol, calibration in IV- and price-space
    (fix `beta`: alpha and beta are not separately identifiable from a single smile)
  - Rough models (Monte Carlo, parallelizable): rBergomi with a Riemann–Liouville driver
    simulated by the hybrid scheme (spot-vol correlation included), and rough Heston
    (Volterra Euler scheme)

- SVI volatility surfaces
  - Raw-SVI per-expiry fits with butterfly no-arbitrage enforced
    (Gatheral–Jacquier `g(k) >= 0` and Roger Lee's wing bound)
  - Calendar stitching (total variance non-decreasing in maturity) and flat-IV
    extrapolation outside the quoted maturities
  - Plots of smiles and surfaces in notebook 15

- Greeks & sensitivities
  - Delta, Gamma, Vega, Theta, Rho (supports dividend yield `q`, including negative `q`)
  - Vanna & Volga

- Strategies and risk
  - Standard strategies (spreads, straddles, strangles, collars, butterflies);
    note `butterfly_spread(S, K1, K2, K3, r, T, sigma)` takes `r` before `T`
  - Payoff diagrams; portfolio aggregation and stress grids
  - VaR/ES (historical and Monte Carlo, one-day horizon; the Monte Carlo methods include one day of
    theta carry, historical VaR shocks the spot only) and P&L attribution

- Barriers and digitals
  - Barrier pricing via MC with Brownian-bridge crossing probabilities, checked against
    Reiner–Rubinstein closed forms
  - Digital cash and asset binaries under Black–Scholes

- Data & utilities
  - `yfinance` helpers for stock/chain data (see `data/`; needs network access)
  - Time-to-maturity, rolling vol, calendars, and helpers
  - Hurst exponent estimators (R/S and DFA; R/S is biased upward for small H)

---

## Installation and tests

```bash
pip install -r requirements.txt          # or: pip install -e ".[data,notebooks,dev]"
pytest                                   # full suite
pytest -m "not slow"                     # skip tests marked slow; CI runs this on every push
```

Run the notebooks from the `notebooks/` directory. Notebooks 08, 09, 13, 14 (live cell), 17, 20 and 21
download market data with `yfinance` and need network access. The optional `numba`
extra speeds up the rough Heston simulation.

---

## Notebooks Map

- 00 Theory: CLT, Ito, Black–Scholes
- 01–07 Core demos: Black–Scholes, Binomial, Monte Carlo, Finite Difference, Greeks, Strategies
- 08–10 Backtesting and portfolio risk (uses optional `yfinance`)
- 11 American: LSMC and binomial
- 12 Heston pricing; 13 Heston calibration (forward-based, vega-weighted)
- 14 SABR calibration (IV- and price-space)
- 15 SVI surface fitting and calendar stitching
- 16 Hurst exponent estimators; 17 Delta hedging
- 18 Barriers (MC + Brownian bridge)
- 19 Rough models; 20 Rough calibration
- 21 Multi-maturity calibration (Heston, rBergomi, Rough Heston)
- 99 Parallel calibration benchmark
