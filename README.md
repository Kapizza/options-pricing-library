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
  - Black–Scholes closed form (with dividend yield `q`)
  - Binomial Tree (European & American)
  - Monte Carlo simulation
  - Finite Difference PDE solvers (explicit / implicit / Crank–Nicolson)
  - American options via Longstaff–Schwartz (LSMC)

- Stochastic volatility models
  - Heston via COS method (fast & stable)
  - SABR (Hagan asymptotic IV) with calibration in IV- and price-space
  - Rough models: rBergomi and Rough Heston (MC), parallelizable

- SVI volatility surfaces
  - Arbitrage-aware SVI parameterization and fitting
  - Calendar stitching to reduce cross-maturity arbitrage
  - Visualization utilities (smiles, 2D/3D surfaces, ATM term structure)

- Greeks & sensitivities
  - Delta, Gamma, Vega, Theta, Rho (supports dividend yield `q`)
  - Vanna & Volga; pathwise and finite-difference Greeks for MC

- Strategies and risk
  - Standard strategies (spreads, straddles, collars, calendars)
  - Payoff diagrams; portfolio aggregation and stress grids
  - VaR/ES (historical and Monte Carlo) and P&L attribution

- Barriers and digitals
  - Barrier pricing via MC with Brownian-bridge correction
  - Digital cash and asset binaries under Black–Scholes

- Data & utilities
  - Optional `yfinance` helpers for stock/chain data (see `data/`)
  - Time-to-maturity, rolling vol, calendars, and helpers

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
