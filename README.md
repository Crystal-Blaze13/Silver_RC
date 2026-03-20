# Silver Price Forecasting Pipeline — Indian Market (VMD + Entropy + LASSO + ARIMA/LSTM + Trading)

This repository implements a weekly silver-price forecasting pipeline for the **Indian market**
over a **25-year window (2000–2026)**, replicating the methodology of Liu et al. (2025).

- **Decomposition:** VMD (Variational Mode Decomposition)
- **Complexity split:** Approximate Entropy (low vs high complexity IMFs)
- **Feature selection:** LASSO per IMF
- **Forecasting:** Hybrid ARIMA/LSTM + single-model/decomposition benchmarks
- **Evaluation:** Error metrics + Diebold–Mariano test + trading simulation

### Professor's extensions (v2)
| Extension | Detail |
|-----------|--------|
| 25-year data window | 2000-01-01 → 2026-03-14 |
| Indian market focus | S&P 500 → Nifty 50 (`^NSEI`); DXY → USD/INR (`USDINR=X`) |
| 12 NLP keywords | 12 India-specific silver Google Trends keywords combined into one index |

---

## 1) Project files

### Data collection / preparation
- `download_financial_data.py` — downloads `silver`, `gold`, `brent`, **`nifty50`**, **`usdinr`** from Yahoo Finance (25-year window).
- `fetch_vix.py` — downloads VIX from Yahoo Finance (25-year window).
- `fetch_trends.py` — downloads **12 India-specific silver keywords** from Google Trends via pytrends, combines into `trends_india.csv`.
- `build_master.py` — merges all sources into `master_weekly_prices.csv` with Indian columns.
- `fetch_trends_missing.py` — fetches the 4 Google Trends keywords that failed in the main batch run, normalises them to the anchor series, and overwrites `trends_india.csv` with all 12 keywords.
- `merge_manual_trends.py` — alternative to the above: merges a manually downloaded `trends_missing_batch.csv` (from trends.google.com) into `trends_india.csv` using a weighted average (8 existing + 4 new).
- `merge.py` — legacy merge helper (not used in main pipeline).
- `test.py` — legacy returns dataset builder (not used in main pipeline).

### Main modelling pipeline
- `step1_vmd_decompose.py`
- `step2_entropy.py`
- `step3_lasso.py`
- `step4_models.py`
- `step5_dmtest.py`
- `step6_trading.py`

### Core dataset
- `master_weekly_prices.csv` (columns: `date,silver,gold,brent,usdinr,nifty50,vix,trends_raw`)

---

## 2) Environment setup

```bash
pip install vmdpy pandas numpy matplotlib scipy scikit-learn statsmodels torch yfinance pytrends
```

---

## 3) Full run path (data download + modelling pipeline)

### Step A — Download market data (25 years, Indian market)

```bash
python download_financial_data.py    # silver, gold, brent, nifty50, usdinr
python fetch_vix.py                  # VIX
python fetch_trends.py               # 12 India silver Google Trends keywords → trends_india.csv
python build_master.py               # merge all → master_weekly_prices.csv
```

> `fetch_trends.py` calls the Google Trends API via pytrends. It may take 2–3 minutes due to
> rate-limit back-offs between batches.

### Step B — Modelling pipeline

```bash
python step1_vmd_decompose.py
python step2_entropy.py
python -c "import pandas as pd, numpy as np; n=int(len(pd.read_csv('master_weekly_prices.csv'))*0.8); np.save('n_train.npy', np.array([n])); print('Saved n_train.npy with n_train =', n)"
python step3_lasso.py
python step4_models.py
python step5_dmtest.py
python step6_trading.py
```

---

## 4) Outputs by step

### Step 1 (`step1_vmd_decompose.py`)
- `fig4_silver_price_split.png`
- `fig7_imf_decomposition.png`
- `table5_imf_statistics.csv`
- `imfs.npy`
- `silver_weekly.csv`

### Step 2 (`step2_entropy.py`)
- `fig8_approximate_entropy.png`
- `imf_complexity.csv`

### Step 3 (`step3_lasso.py`)
- `table6_lasso_features.csv`
- `lasso_selected_features.pkl`

### Step 4 (`step4_models.py`)
- `table7_single_model_errors.csv`
- `table8_decomp_model_errors.csv`
- `fig9_single_model_forecasts.png`
- `fig10_error_barplots.png`
- `predictions.pkl`

### Step 5 (`step5_dmtest.py`)
- `table9_dm_test.csv`

### Step 6 (`step6_trading.py`)
- `table10_interval_errors.csv`
- `table11_decomp_trading.csv`
- `table12_single_trading.csv`
- `fig11_interval_forecasts.png`
- `fig12_trading_strategy_illustration.png`
- `fig13_trading_evaluation.png`

---

## 5) Optional: refresh raw market data

If you want to re-download raw series before running modelling:

```bash
python download_financial_data.py
python fetch_vix.py
```

This updates files inside `financial_data/` and `vix.csv`.

---

## 6) Notes / caveats

- Run all scripts from the repository root.
- Step 4 can be slower because of repeated ARIMA fits and LSTM training.
- Current benchmark names include “CEEMDAN-*” approximations generated from VMD outputs + noise (as coded in `step4_models.py`).
- If you regenerate datasets yourself, ensure `master_weekly_prices.csv` keeps the expected columns:
  - `silver, gold, brent, usdinr, nifty50, vix, trends_raw`

---

## 7) Quick command block (full pipeline)

```bash
# Data (run once)
python download_financial_data.py
python fetch_vix.py
python fetch_trends.py
python build_master.py

# Modelling
python step1_vmd_decompose.py
python step2_entropy.py
python -c "import pandas as pd, numpy as np; n=int(len(pd.read_csv('master_weekly_prices.csv'))*0.8); np.save('n_train.npy', np.array([n])); print('Saved n_train.npy with n_train =', n)"
python step3_lasso.py
python step4_models.py
python step5_dmtest.py
python step6_trading.py
```
