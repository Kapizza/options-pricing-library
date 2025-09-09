# 📈 Options Pricing Library

A **Python library for option pricing, Greeks, strategies, volatility modeling, risk management, and backtesting**, with clean implementations of classic models and a full suite of demo notebooks.  

This project is designed as both an educational toolkit and a reusable research library for quantitative finance.


## Theory Reference

For detailed explanations of the fundamental theories used in this package, including the **Central Limit Theorem (CLT)**, **Brownian Motion**, **Itô’s Lemma**, and the **Black–Scholes Model**:

**[`notebooks/00_Theory-CLT-ITO-BS.ipynb`](notebooks/00_Theory-CLT-ITO-BS.ipynb)**

---

## 🚀 Features

- **Core Pricing Models**
  - Black–Scholes closed form
  - Binomial Tree (European & American)
  - Monte Carlo simulation
  - Finite Difference PDE solvers (explicit / implicit / Crank–Nicolson)

- **Greeks & Higher-Order Sensitivities**
  - Δ, Γ, Vega, Θ, ρ
  - Vanna & Volga

- **Option Strategies**
  - Vanilla legs (long/short calls & puts)
  - Spreads (bull, bear, butterfly)
  - Combinations (straddle, strangle, collar, covered call, calendars)

- **Volatility Tools**
  - Implied volatility solver & surfaces
  - Moneyness grids and visualization

- **Risk & Portfolio Analytics**
  - Aggregated Greeks
  - Scenario analysis (Taylor vs full revaluation)
  - P&L attribution
  - Stress testing grids
  - Historical & Monte Carlo VaR/ES

- **Backtesting**
  - Constant σ vs realized σ comparisons
  - Strategy rolling backtests

- **Utilities**
  - Roll calendars (monthly/weekly)
  - Realized volatility estimators
  - Time-to-maturity helpers


