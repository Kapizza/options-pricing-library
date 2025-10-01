# Options Pricing Library

A **Python library for option pricing, Greeks, strategies, volatility modeling, risk management, calibration, and backtesting**, with clean implementations of classic models and a full suite of demo notebooks.  

This project is designed as an educational and demonstrational toolkit for quantitative finance.

---

## Theory Reference

For detailed explanations of the fundamental theories used in this package, including the **Central Limit Theorem (CLT)**, **Brownian Motion**, **Itô’s Lemma**, and the **Black–Scholes Model**:

**[`notebooks/00_Theory-CLT-ITO-BS.ipynb`](notebooks/00_Theory-CLT-ITO-BS.ipynb)**

---

## Features

- **Core Pricing Models**
  - Black–Scholes closed form (with dividend yield `q`)
  - Binomial Tree (European & American)
  - Monte Carlo simulation
  - Finite Difference PDE solvers (explicit / implicit / Crank–Nicolson)
  - American options via Longstaff–Schwartz (LSMC)

- **Stochastic Volatility Models**
  - Heston model via **COS method** (fast & stable)
  - Batch multi-strike Heston pricer (vectorized COS)
  - SABR model (Hagan asymptotic IV)
  - Calibration routines (price- and IV-space, vega-weighted, forward-based)
  - Synthetic surface generation & recovery tests

- **SVI Volatility Surfaces**
  - Arbitrage-free SVI parameterization
  - Calibration from implied vols or option mid prices
  - **Calendar projection** to enforce no-arbitrage across maturities
  - Visualization: smiles, 2D/3D IV surfaces, ATM term structure
  - Surface archiving & replay

- **Greeks & Higher-Order Sensitivities**
  - Δ, Γ, Vega, Θ, ρ (all with dividend yield `q`)
  - Vanna & Volga
  - Pathwise & finite-difference Greeks for Monte Carlo

- **Option Strategies**
  - Vanilla legs (long/short calls & puts)
  - Spreads (bull, bear, butterfly)
  - Combinations (straddle, strangle, collar, covered call, calendars)
  - Payoff diagram visualizations

- **Risk & Portfolio Analytics**
  - Aggregated Greeks
  - Scenario analysis (Taylor vs full revaluation)
  - P&L attribution (Greeks vs realized)
  - Stress testing grids
  - Historical & Monte Carlo VaR/ES

- **Backtesting**
  - Strategy rolling backtests
  - Constant σ vs realized σ comparisons
  - **Delta-hedging backtests** with frozen surfaces  
    - Sticky-moneyness vs sticky-strike dynamics  
    - PnL attribution and transaction-cost extensions
  - Free-float weighted equity indices
  - Return correlation heatmaps

- **Volatility & Time Series Tools**
  - Implied volatility solvers (Brent + Newton polish)
  - Surface construction (strike, maturity, moneyness)
  - Volatility smile fitting and diagnostics
  - Realized volatility estimators (rolling, EWMA, GARCH-ready)
  - **Hurst exponent estimators** (R/S and DFA methods)

- **Data Utilities**
  - `yfinance` integration for stock data & option chains
  - Forward/discount inference via put–call parity regression
  - Realized volatility and variance swaps
  - Time-to-maturity helpers
  - Calendar rolling utilities

---
