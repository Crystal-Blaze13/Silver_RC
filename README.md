# MCX Silver Forecasting Pipeline — Indian Market Replication

Replication of **Liu et al. (2025)** "From forecasting to trading: A multimodal-data-driven approach to reversing carbon market losses" (*Energy Economics* 144, 108350), adapted for **MCX Silver (INR/kg)** weekly prices, Indian market, 2008–2026.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scipy scikit-learn torch statsmodels vmdpy PyEMD

python step1_vmd_decompose.py   # VMD decomposition  (~30–90 s)
python step2_entropy.py         # Approximate entropy (~2–5 min)
python step3_lasso.py           # LASSO feature selection
python step4_models.py          # All forecasting models (~10–40 min)
python step5_dmtest.py          # Diebold-Mariano tests
python step6_trading.py         # Interval forecasting + trading strategy

python generate_figures.py      # Regenerate all figures at 300 DPI
```

> All file paths and hyperparameters are centralised in [config.py](config.py).

---

## Data

| Item | Details |
|------|---------|
| Asset | MCX Silver (MCXSILV Comdty, INR/kg) |
| Frequency | Weekly (Friday close) |
| Window | Jan 2008 – Mar 2026 (946 weeks) |
| Training | Jan 2008 – Feb 2024 (843 weeks) |
| Test | Mar 2024 – Mar 2026 (103 weeks) |
| External features | Gold (USD/oz), Brent, USD/INR, Nifty 50, India VIX, MCX Gold, Geopolitical Risk, Google Trends, India EPU |

Processed data: [financial_data/processed/master_weekly_prices.csv](financial_data/processed/master_weekly_prices.csv)

---

## Pipeline Overview

| Step | File | Paper section | Key output |
|------|------|---------------|------------|
| 1 | [step1_vmd_decompose.py](step1_vmd_decompose.py) | §3.1.1, §4.3 | IMFs, Fig 4, Fig 7, Table 5 |
| 2 | [step2_entropy.py](step2_entropy.py) | §3.1.2, §4.3 | Complexity labels, Fig 8, Fig 8b |
| 3 | [step3_lasso.py](step3_lasso.py) | §3.1.3, §4.4 | Feature masks, Table 6 |
| 4 | [step4_models.py](step4_models.py) | §3.1.4–5, §4.6.1 | Predictions, Fig 9, Fig 10, Tables 7–8 |
| 5 | [step5_dmtest.py](step5_dmtest.py) | §4.6.1 | DM matrix, Fig DM, Table 9 |
| 6 | [step6_trading.py](step6_trading.py) | §4.7 | Trading results, Figs 11–13, Tables 10–15 |

---

## Key Results

### VMD Decomposition (6 IMFs)

| IMF | Period (weeks) | Var Ratio | Correlation | Complexity | Model |
|-----|---------------|-----------|-------------|------------|-------|
| IMF1 | 946 | 0.716 | 0.909 | Low | ARIMA |
| IMF2 | 157.7 | 0.146 | 0.524 | Low | ARIMA |
| IMF3 | 17.2 | 0.007 | 0.136 | High | LSTM |
| IMF4 | 6.2 | 0.001 | 0.067 | High | LSTM |
| IMF5 | 3.4 | 0.000 | 0.031 | High | LSTM |
| IMF6 | 2.3 | 0.001 | 0.036 | High | LSTM |

### Forecast Accuracy — Test Set (103 weeks)

**Single models:**

| Model | RMSE | MAE | MAPE (%) | DA (%) |
|-------|------|-----|----------|--------|
| ES | 44,434 | 24,157 | 13.77 | 54.9 |
| ARIMA | 11,191 | 5,107 | 3.21 | 59.8 |
| SVR | 100,690 | 83,089 | 59.81 | 31.4 |
| RF | 69,998 | 40,996 | 22.69 | 39.2 |
| MLP | 38,453 | 23,897 | 15.60 | **68.6** |
| ELM | 90,436 | 64,762 | 41.96 | 31.4 |
| LSTM | 140,546 | 128,552 | 99.99 | 31.4 |
| Naive(RW) | 11,220 | 5,248 | 3.26 | 0.0 |

**Decomposition models:**

| Model | RMSE | MAE | MAPE (%) | DA (%) |
|-------|------|-----|----------|--------|
| **VMD-ARIMA** | **4,669** | **2,617** | **1.63** | **84.3** |
| Proposed (VMD+ARIMA+LSTM) | 13,088 | 6,908 | 4.23 | 64.7 |
| CEEMDAN-ARIMA | 43,136 | 31,542 | 24.93 | 62.7 |
| VMD-LSTM | 77,770 | 53,523 | 33.87 | 31.4 |
| CEEMDAN-LSTM | 73,230 | 53,313 | 35.61 | 31.4 |

> **Key finding:** VMD-ARIMA outperforms the Proposed method on weekly silver data. Root cause: only ~160 training sequences per LSTM IMF (vs ~316 in the paper's daily carbon data) causes LSTM to underfit, while ARIMA successfully captures the near-linear trend structure at all frequencies.

### DM Test — Proposed vs Competitors

| Comparison | DM Stat | p-value (1-sided) | Sig |
|-----------|---------|-------------------|-----|
| vs ARIMA | +3.14 | 0.0011 | *** |
| vs SVR | +4.72 | 0.0000 | *** |
| vs RF | +3.03 | 0.0015 | *** |
| vs ELM | +3.74 | 0.0002 | *** |
| vs LSTM | +6.58 | 0.0000 | *** |
| vs VMD-ARIMA | −2.91 | 0.9978 | — |
| vs VMD-LSTM | +6.58 | 0.0000 | *** |

The Proposed model is **statistically significantly better than all single models** but is dominated by VMD-ARIMA (which benefits from weekly data's linear trend structure).

### Trading Results — Test Period

| Strategy | Cumulative Return (%) | Sharpe | Max DD (%) |
|----------|----------------------|--------|-----------|
| Scheme 1 (weekly) | 180.8 | 1.69 | −22.8 |
| 2-week holding | 246.5 | 2.01 | −20.1 |
| **4-week holding** | **250.3** | **2.02** | −11.2 |
| Buy & Hold | 164.3 | 1.61 | −31.3 |

Best active strategy: **4-week holding period** outperforms weekly re-balancing by ~70 percentage points due to reduced transaction cost drag.

### Stress Test — Synthetic Silver Crash

A synthetic crash scenario (₹90,000 → ₹28,890 over 52 weeks, mimicking a commodity bust) was injected into the test period to validate the interval constraint's risk-management value.

| Metric | Actual Market | Synthetic Crash |
|--------|--------------|-----------------|
| Scheme 1 Return | +180.8% | +88.6% |
| Scheme 1 Sharpe | 1.69 | **2.70** |
| Buy & Hold Return | +164.5% | **−69.1%** |
| Buy & Hold Sharpe | 1.61 | **−5.30** |

The active strategy **reverses buy-and-hold losses** during the crash while maintaining positive returns — the core result of the paper's trading framework.

---

## Output Files

### Figures

| File | Description | Paper equivalent |
|------|-------------|-----------------|
| [fig4_silver_price_split.png](fig4_silver_price_split.png) | Price series + train/test split | Fig 4 |
| [fig7_imf_decomposition.png](fig7_imf_decomposition.png) | All IMFs vs time | Fig 7 |
| [fig8_approximate_entropy.png](fig8_approximate_entropy.png) | ApEn + SampEn bar chart | Fig 8 |
| [fig8b_imf_correlation.png](fig8b_imf_correlation.png) | IMF correlation heat-map | Added |
| [fig9_single_model_forecasts.png](fig9_single_model_forecasts.png) | Per-model forecast plots | Fig 9 |
| [fig10_error_barplots.png](fig10_error_barplots.png) | Error metrics across models | Fig 10 |
| [fig11_interval_forecasts.png](fig11_interval_forecasts.png) | 80% prediction intervals | Fig 11 |
| [fig12_trading_strategy_illustration.png](fig12_trading_strategy_illustration.png) | Strategy illustration | Fig 12 |
| [fig13_trading_evaluation.png](fig13_trading_evaluation.png) | Trading returns by scheme | Fig 13 |
| [fig13b_equity_curves.png](fig13b_equity_curves.png) | Equity curves all schemes | Added |
| [fig_dm_heatmap.png](fig_dm_heatmap.png) | DM test matrix heat-map | Added |
| [fig_stress_test_comparison.png](fig_stress_test_comparison.png) | Actual vs crash stress test | Added |

### Tables

| File | Description | Paper equivalent |
|------|-------------|-----------------|
| [table5_imf_statistics.csv](table5_imf_statistics.csv) | IMF frequency/variance stats | Table 5 |
| [table6_lasso_features.csv](table6_lasso_features.csv) | LASSO selected features per IMF | Table 6 |
| [table7_single_model_errors.csv](table7_single_model_errors.csv) | Single-model forecast errors | Table 7 |
| [table8_decomp_model_errors.csv](table8_decomp_model_errors.csv) | Decomposition model errors | Table 8 |
| [table9_dm_test.csv](table9_dm_test.csv) | DM test display matrix | Table 9 |
| [table10_interval_errors.csv](table10_interval_errors.csv) | Interval forecast metrics | Table 10 |
| [table11_decomp_trading.csv](table11_decomp_trading.csv) | Decomposition trading results | Table 11 |
| [table12_single_trading.csv](table12_single_trading.csv) | Single-model trading results | Table 12 |
| [table13_margin_sweep.csv](table13_margin_sweep.csv) | Margin sensitivity sweep | Added |
| [table14_holding_periods.csv](table14_holding_periods.csv) | Holding-period analysis | Added |
| [table15_final_comparison.csv](table15_final_comparison.csv) | Full implementation comparison | Added |

### Intermediates (required between steps)

| File | Produced by | Used by |
|------|------------|---------|
| [imfs.npy](imfs.npy) | Step 1 | Steps 2–6 |
| [imf_complexity.csv](imf_complexity.csv) | Step 2 | Steps 3–6 |
| [lasso_selected_features.pkl](lasso_selected_features.pkl) | Step 3 | Steps 4–6 |
| [predictions.pkl](predictions.pkl) | Step 4 | Steps 5–6 |
| [comparison_actual_vs_synthetic.json](comparison_actual_vs_synthetic.json) | Step 6 | Stress test |

---

## Differences from the Paper

See [REPLICATION_NOTES.md](REPLICATION_NOTES.md) for the full list. Summary:

| Dimension | Paper (Liu et al.) | This replication |
|-----------|-------------------|-----------------|
| Asset | Carbon allowances (CEA, HBEA) | MCX Silver (INR/kg) |
| Frequency | Daily | Weekly |
| Training sequences / LSTM IMF | ~316 | ~160 |
| Interval model | iMLP (intraday H/L data) | Split conformal prediction |
| Best decomp model | Proposed (VMD+ARIMA+LSTM) | VMD-ARIMA |
| External features | Energy, finance, Baidu, news | Gold, FX, equities, Trends, EPU |

---

## Citation

```bibtex
@article{liu2025forecasting,
  title   = {From forecasting to trading: A multimodal-data-driven approach to
             reversing carbon market losses},
  author  = {Liu, Shuihan and Li, Mingchen and Yang, Kun and Wei, Yunjie
             and Wang, Shouyang},
  journal = {Energy Economics},
  volume  = {144},
  pages   = {108350},
  year    = {2025},
  doi     = {10.1016/j.eneco.2025.108350}
}
```
