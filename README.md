# Silver Price Forecasting — Weekly vs Daily Comparison

VMD-based hybrid ARIMA/LSTM model for Indian silver (MCX, INR/kg), evaluated at two frequencies.

---

## Pipeline Overview

Both pipelines share the same architecture:
1. **VMD decomposition** — split silver price into IMFs
2. **Entropy classification** — route low-complexity IMFs to ARIMA, high-complexity to LSTM
3. **LASSO feature selection** — per-IMF feature selection from price lags + exogenous variables
4. **Hybrid forecasting** — ARIMA + LSTM per IMF, sum for final prediction
5. **Interval forecasting** — conformal (weekly) or iMLP (daily) uncertainty quantification
6. **Trading evaluation** — 4 scheme variants with/without interval-gating

---

## Key Differences

| Property | Weekly | Daily |
|----------|--------|-------|
| Frequency | W-SUN | Daily |
| IMFs extracted | 7 | 8 |
| Dominant IMF period | ~946 weeks (18 yrs) | ~2,930 days (8 yrs) |
| IMF1 variance ratio | 71.6% | 93.6% |
| Low-complexity IMFs | IMF1, IMF2, IMF7 | IMF1, IMF2 |
| High-complexity IMFs | IMF3–IMF6 | IMF3–IMF8 |
| Price lags in LASSO | 5 | 20 |
| Technical indicators | No | Yes (RSI, MA ratio, vol, rolling rets) |
| Avg features per IMF | ~10 | ~33 |
| Interval method | Conformal prediction | iMLP |

---

## Forecast Accuracy — Proposed Model

| Metric | Weekly | Daily |
|--------|--------|-------|
| RMSE | 13,088 | 3,628 |
| MAE | 6,908 | 1,593 |
| MAPE (%) | 4.23 | 0.92 |
| DA (%) | 64.7 | 47.4 |

Daily resolution substantially improves error metrics (MAPE 0.92% vs 4.23%). Directional accuracy is lower daily because noise-to-signal ratio is higher at finer granularity.

### Best decomposition model (VMD-ARIMA)

| Metric | Weekly | Daily |
|--------|--------|-------|
| RMSE | 4,669 | 2,131 |
| MAE | 2,617 | 1,009 |
| MAPE (%) | 1.63 | 0.58 |
| DA (%) | 84.3 | 51.9 |

VMD-ARIMA achieves sub-1% MAPE at daily resolution. The weekly pipeline benefits more from directional accuracy (84.3%) while daily achieves finer absolute error.

---

## Forecast Accuracy — Single Models (No Decomposition)

| Model | Weekly RMSE | Daily RMSE | Weekly MAPE | Daily MAPE |
|-------|------------|------------|-------------|------------|
| ARIMA | 11,190 | 5,952 | 3.21% | 1.11% |
| Naive (RW) | 11,220 | 5,952 | 3.26% | 1.11% |
| MLP | 38,453 | 547,306 | 15.60% | 102.4% |
| LSTM | 140,546 | 69,666 | 99.99% | 22.4% |

ARIMA without decomposition performs comparably to a random walk at both frequencies — decomposition is essential for improvement.

---

## Trading Performance

### Scheme 1 — Trade Every Signal (Unconstrained)

| Model | Weekly Cum. Ret. | Weekly Sharpe | Daily Cum. Ret. | Daily Sharpe |
|-------|-----------------|---------------|-----------------|--------------|
| VMD-ARIMA | 1,370.6% | 4.82 | 22,980.4% | 7.66 |
| Proposed | 180.8% | 1.69 | 11,507.1% | 6.50 |
| CEEMDAN-ARIMA | 63.3% | 0.88 | 96.0% | 1.01 |

Daily cumulative returns are orders of magnitude larger due to daily compounding (102 weekly vs ~182 daily trades, but daily captures many more short-term movements).

### Scheme 1' — Interval-Gated (Proposed Model Only)

| Metric | Weekly | Daily |
|--------|--------|-------|
| Cumulative return | 115.4% | 4,501.5% |
| Sharpe (annualised) | 1.56 | 5.46 |
| Directional accuracy | 72.0% | 94.9% |
| Profit factor | 4.38 | 43.7 |
| Max drawdown | -11.2% | -5.6% |
| Trades taken | 25 / 102 (24.5%) | 136 / 310 (43.9%) |
| Binomial p-value | 0.022 | 0.000 |

The daily interval gate (iMLP) is far more discriminating — 94.9% hit rate on traded days vs 72% weekly. This reflects tighter, better-calibrated uncertainty bounds at daily resolution.

---

## Interval Forecasting

| Metric | Weekly (Conformal) | Daily (iMLP) |
|--------|--------------------|--------------|
| Method | Conformal prediction | iMLP |
| Coverage | 69.6% (target: 70%) | CR = 0.035 (test) |
| Avg width | 0.1045 | ARV = 0.136 |

Weekly conformal prediction achieves near-exact target coverage (69.6% vs 70%). Daily iMLP shows ARV < 1 on test set, indicating intervals improve on a naive benchmark.

---

## Structure

```
rc/
├── common_data/            # shared raw inputs (commodity CSVs, Bloomberg xlsx, GDELT, trends)
├── silver/
│   ├── code/               # silver daily pipeline scripts
│   ├── data/               # silver intermediates (master_daily_prices.csv, imfs_daily.npy, etc.)
│   └── results/
│       ├── figures/        # silver plots
│       └── tables/         # silver result CSVs and JSON
└── gold_daily_code/
    ├── code/               # gold daily pipeline scripts
    ├── data/               # gold intermediates
    └── results/
        ├── figures/        # gold plots
        └── tables/         # gold result CSVs and JSON
```

See [silver/README.md](silver/README.md) for silver pipeline detail.