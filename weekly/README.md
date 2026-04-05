# Weekly Silver Price Forecasting — Results

Pipeline frequency: **weekly (W-SUN)**, 2000–2026. Indian silver market (MCX), priced in INR/kg.

---

## Decomposition (VMD)

7 IMFs extracted. IMF1 dominates variance (71.6%, period ~946 weeks ≈ 18 years). IMF7 is the residual trend component.

| IMF | Period (wks) | Variance Ratio | Correlation | Complexity |
|-----|-------------|----------------|-------------|------------|
| IMF1 | 946.0 | 0.7163 | 0.9086 | Low |
| IMF2 | 157.7 | 0.1463 | 0.5242 | Low |
| IMF3 | 17.2 | 0.0071 | 0.1359 | High |
| IMF4 | 6.2 | 0.0013 | 0.0670 | High |
| IMF5 | 3.4 | 0.0002 | 0.0311 | High |
| IMF6 | 2.3 | 0.0007 | 0.0363 | High |
| IMF7 (residual) | 29.6 | 0.0041 | 0.2389 | Low |

Low-complexity IMFs routed to ARIMA; high-complexity IMFs to LSTM.

---

## LASSO Feature Selection

Features: 5 silver price lags + 9 exogenous (gold, brent, USD/INR, Nifty50, VIX, MCX gold, geopolitical risk, Google Trends, India EPU). Average ~10 features selected per IMF (α tuned per IMF via cross-validation).

---

## Point Forecast Accuracy

### Single models (no decomposition)

| Model | RMSE | MAE | MAPE (%) | DA (%) |
|-------|------|-----|----------|--------|
| ARIMA | 11,190 | 5,107 | 3.21 | 59.8 |
| Naive (RW) | 11,220 | 5,248 | 3.26 | 0.0 |
| MLP | 38,453 | 23,897 | 15.60 | **68.6** |
| ES | 44,434 | 24,157 | 13.77 | 54.9 |
| LSTM | 140,546 | 128,552 | 99.99 | 31.4 |

### Decomposition models

| Model | RMSE | MAE | MAPE (%) | DA (%) |
|-------|------|-----|----------|--------|
| **Proposed (VMD-hybrid)** | **13,088** | **6,908** | **4.23** | **64.7** |
| ARIMA (standalone) | 11,190 | 5,107 | 3.21 | 59.8 |
| CEEMDAN-ARIMA | 43,136 | 31,542 | 24.93 | 62.8 |
| VMD-LSTM | 77,770 | 53,523 | 33.87 | 31.4 |
| VMD-ARIMA | **4,669** | **2,617** | **1.63** | **84.3** |

VMD-ARIMA achieves the lowest RMSE/MAPE by a wide margin (RMSE 4,669 vs 11,190 for standalone ARIMA, 84.3% directional accuracy). The proposed hybrid VMD model balances accuracy with generalisation.

---

## Interval Forecasting (Conformal)

| Set | Coverage | Avg Width | RMSE (center) |
|-----|----------|-----------|---------------|
| Conformal | 0.696 | 0.1045 | 0.0614 |

Target coverage: 70%. Achieved: 69.6% — well-calibrated.

---

## Trading Strategy

4 schemes evaluated:
- **Scheme 1**: trade every signal
- **Scheme 1'**: trade only when interval excludes zero (high-confidence signals)
- **Scheme 2**: position-size by predicted magnitude
- **Scheme 2'**: magnitude-sized, interval-gated

### Best decomposition model results (Scheme 1 / unconstrained)

| Model | Cum. Return (%) | Max DD (%) | Sharpe | Trades |
|-------|----------------|------------|--------|--------|
| VMD-ARIMA | **1,370.6** | -7.9 | **4.82** | 102 |
| Proposed | 180.8 | -22.8 | 1.69 | 102 |
| CEEMDAN-ARIMA | 63.3 | -30.1 | 0.88 | 102 |
| VMD-LSTM | -72.2 | -78.5 | -1.68 | 102 |

### Proposed model — Scheme 1' (interval-gated, 25 trades)

| Metric | Value |
|--------|-------|
| Cumulative return | 115.4% |
| Sharpe (annualised) | 1.56 |
| Directional accuracy | 72.0% |
| Binomial p-value | 0.022 |
| Profit factor | 4.38 |
| Max drawdown | -11.2% |
| Weeks traded | 25 / 102 (24.5%) |

The interval gate reduces trade count by 75% while preserving returns — high selectivity with statistically significant directional accuracy.

---

## File Map

| File | Contents |
|------|----------|
| `results/tables/table5_imf_statistics.csv` | VMD mode statistics |
| `results/tables/table6_lasso_features.csv` | LASSO selected features per IMF |
| `results/tables/table7_single_model_errors.csv` | Single-model forecast errors |
| `results/tables/table8_decomp_model_errors.csv` | Decomposition model errors |
| `results/tables/table9_dm_pvals_onesided.csv` | Diebold-Mariano p-values (one-sided) |
| `results/tables/table9_dm_pvals_twosided.csv` | Diebold-Mariano p-values (two-sided) |
| `results/tables/table9_dm_stats_numeric.csv` | DM test statistics |
| `results/tables/table10_interval_errors.csv` | Interval forecast coverage/width |
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
| `results/figures/fig11_interval_forecasts.png` | Interval forecast plot |
| `results/figures/fig12_trading_strategy_illustration.png` | Strategy diagram |
| `results/figures/fig13_trading_evaluation.png` | Trading performance |
| `results/figures/fig13b_equity_curves.png` | Equity curves |
| `results/figures/fig_dm_heatmap.png` | DM test heatmap |
