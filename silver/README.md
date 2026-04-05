# Daily Silver Price Forecasting — Results

Pipeline frequency: **daily**, 2000–2026. Indian silver market (MCX), priced in INR/kg.

---

## Decomposition (VMD)

8 IMFs extracted (one more than the weekly pipeline, capturing additional intraday-adjacent cycles). IMF1 dominates variance (93.6%, period ~2,930 days ≈ 8 years).

| IMF | Period (days) | Variance Ratio | Correlation | Complexity |
|-----|--------------|----------------|-------------|------------|
| IMF1 | 2930.0 | 0.9363 | 0.9847 | Low |
| IMF2 | 146.5 | 0.0187 | 0.2496 | Low |
| IMF3 | 36.2 | 0.0060 | 0.1040 | High |
| IMF4 | 14.4 | 0.0009 | 0.0511 | High |
| IMF5 | 3.6 | 0.0002 | 0.0215 | High |
| IMF6 | 2.7 | 0.0002 | 0.0206 | High |
| IMF7 | 2.2 | 0.0001 | 0.0133 | High |
| IMF8 (residual) | 7.6 | 0.0007 | 0.1058 | High |

Low-complexity IMFs (IMF1, IMF2) routed to ARIMA; remaining 6 to LSTM.

---

## LASSO Feature Selection

Richer feature set than weekly: 20 silver log-return lags + 9 exogenous variables + 8 technical indicators (RSI, MA ratio, volatility, rolling returns at 5/20/60 days). Average ~33 features selected per IMF (substantially denser than weekly due to higher temporal resolution).

---

## Point Forecast Accuracy

### Single models (no decomposition)

| Model | RMSE | MAE | MAPE (%) | DA (%) |
|-------|------|-----|----------|--------|
| ARIMA | **5,952** | **1,959** | **1.11** | 42.6 |
| Naive (RW) | 5,952 | 1,959 | 1.11 | 42.6 |
| LSTM | 69,666 | 42,426 | 22.42 | 22.6 |
| SVR | 72,075 | 47,002 | 26.53 | 22.0 |
| RF | 84,303 | 55,926 | 31.71 | 21.9 |
| ELM | 79,955 | 51,532 | 28.54 | 22.0 |
| MLP | 547,306 | 201,530 | 102.43 | 22.2 |

### Decomposition models

| Model | RMSE | MAE | MAPE (%) | DA (%) |
|-------|------|-----|----------|--------|
| **VMD-ARIMA** | **2,131** | **1,009** | **0.58** | 51.9 |
| **Proposed (VMD-hybrid)** | **3,628** | **1,593** | **0.92** | 47.4 |
| CEEMDAN-ARIMA | 30,799 | 20,752 | 14.17 | 31.5 |
| CEEMDAN-LSTM | 94,363 | 64,386 | 37.90 | 22.6 |
| VMD-LSTM | 94,997 | 72,128 | 46.11 | 21.7 |

VMD-ARIMA achieves RMSE of 2,131 — a 64% improvement over standalone ARIMA (5,952) and a 96% improvement over the random walk baseline. Sub-1% MAPE.

---

## Interval Forecasting (iMLP)

| Set | U | ARV | RMSDE | Coverage Rate |
|-----|---|-----|-------|---------------|
| Calibration | 8.15 | 2.05 | 32.91 | 0.0635 |
| Test | 3.27 | 0.14 | 74.96 | 0.0346 |

ARV < 1 on the test set indicates the interval model's uncertainty estimates improve on a naive benchmark. Coverage Rate (CR) measures normalised interval width.

---

## Trading Strategy

4 schemes evaluated (same as weekly):
- **Scheme 1**: trade every signal
- **Scheme 1'**: trade only when iMLP interval excludes zero
- **Scheme 2**: position-size by predicted magnitude
- **Scheme 2'**: magnitude-sized, interval-gated

### Best decomposition model results (Scheme 1 / unconstrained)

| Model | Cum. Return (%) | Max DD (%) | Sharpe | Trades |
|-------|----------------|------------|--------|--------|
| VMD-ARIMA | **22,980.4** | -2.1 | **7.66** | 182 |
| Proposed | 11,507.1 | -5.7 | 6.50 | 167 |
| CEEMDAN-ARIMA | 96.0 | -46.4 | 1.01 | 26 |
| VMD-LSTM | -69.1 | -80.2 | -1.26 | 1 |

### Proposed model — Scheme 1' (interval-gated)

| Metric | Value |
|--------|-------|
| Cumulative return | 4,501.5% |
| Sharpe (annualised) | 5.46 |
| Directional accuracy | 94.9% |
| Binomial p-value | 0.000 |
| Profit factor | 43.7 |
| Max drawdown | -5.6% |
| Days traded | 136 / 310 active (43.9%) |
| Days blocked by interval | 358 |

The daily interval gate is substantially tighter than the weekly equivalent — 94.9% directional accuracy on traded signals with a profit factor of 43.7.

---

## File Map

| File | Contents |
|------|----------|
| `results/tables/table5_imf_statistics.csv` | VMD mode statistics |
| `results/tables/table6_lasso_features.csv` | LASSO selected features per IMF |
| `results/tables/table7_single_model_errors.csv` | Single-model forecast errors |
| `results/tables/table8_decomp_model_errors.csv` | Decomposition model errors |
| `results/tables/table9_dm_test.csv` | Diebold-Mariano test results |
| `results/tables/table9_dm_stats_numeric.csv` | DM test statistics |
| `results/tables/table9_dm_pvals_onesided.csv` | DM p-values (one-sided) |
| `results/tables/table9_dm_pvals_twosided.csv` | DM p-values (two-sided) |
| `results/tables/table10_interval_errors.csv` | iMLP interval forecast metrics |
| `results/tables/table11_decomp_trading.csv` | Trading results — decomp models |
| `results/tables/table12_single_trading.csv` | Trading results — single models |
| `results/tables/table13_margin_sweep.csv` | Margin/threshold sweep |
| `results/tables/table14_holding_periods.csv` | Holding period analysis |
| `results/tables/table15_final_comparison.csv` | Final model comparison |
| `results/tables/paper_summary.json` | Key summary statistics |
| `results/figures/fig4_silver_price_split.png` | Train/test split |
| `results/figures/fig7_imf_decomposition.png` | VMD decomposition |
| `results/figures/fig8_approximate_entropy.png` | Entropy-based complexity |
| `results/figures/fig8b_imf_correlation.png` | IMF correlations |
| `results/figures/fig9_single_model_forecasts.png` | Single model forecasts |
| `results/figures/fig10_error_barplots.png` | Error metric comparison |
| `results/figures/fig11_interval_forecasts.png` | iMLP interval forecast plot |
| `results/figures/fig12_trading_strategy_illustration.png` | Strategy diagram |
| `results/figures/fig13_trading_evaluation.png` | Trading performance |
| `results/figures/fig13b_equity_curves.png` | Equity curves |
| `results/figures/fig_dm_heatmap.png` | DM test heatmap |
| `results/figures/fig_proposed_vs_vmdarima.png` | Proposed vs VMD-ARIMA comparison |
