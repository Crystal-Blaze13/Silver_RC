# Silver Price Forecasting Pipeline (VMD + Entropy + LASSO + ARIMA/LSTM + Trading)

This repository implements a weekly silver-price forecasting pipeline and evaluation workflow:

- **Decomposition:** VMD (Variational Mode Decomposition)
- **Complexity split:** Approximate Entropy (low vs high complexity IMFs)
- **Feature selection:** LASSO per IMF
- **Forecasting:** Hybrid ARIMA/LSTM + single-model/decomposition benchmarks
- **Evaluation:** Error metrics + Diebold–Mariano test + trading simulation

---

## 1) Project files

### Data collection / preparation
- `download_financial_data.py` — downloads market series (`silver`, `gold`, `brent`, `dxy`, `sp500`) from Yahoo Finance.
- `fetch_vix.py` — downloads VIX from Yahoo Finance.
- `merge.py` — loads files from `financial_data/` and merges by date (currently prints merged shape/head).
- `test.py` — builds a weekly **returns** dataset from prices + Google Trends (`merged_weekly_dataset.csv`).

### Main modelling pipeline
- `step1_vmd_decompose.py`
- `step2_entropy.py`
- `step3_lasso.py`
- `step4_models.py`
- `step5_dmtest.py`
- `step6_trading.py`

### Existing core dataset
- `master_weekly_prices.csv` (columns: `date,silver,gold,brent,dxy,sp500,vix,trends_raw`)

---

## 2) Environment setup

```bash
pip install vmdpy pandas numpy matplotlib scipy scikit-learn statsmodels torch yfinance
```

> You already installed the key dependencies, so this is just for reproducibility.

---

## 3) Recommended run path (using existing `master_weekly_prices.csv`)

If you only want to run the paper-style modelling pipeline:

```bash
python step1_vmd_decompose.py
python step2_entropy.py
```

### Important: create `n_train.npy` (required by Steps 3–4)

`step3_lasso.py` and `step4_models.py` expect `n_train.npy`, but it is not generated automatically by the current scripts.

Run this once after Step 1:

```bash
python -c "import pandas as pd, numpy as np; n=int(len(pd.read_csv('master_weekly_prices.csv'))*0.8); np.save('n_train.npy', np.array([n])); print('Saved n_train.npy with n_train =', n)"
```

Then continue:

```bash
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

- Use the scripts from repository root (`/Users/palakkshetrapal/rc`).
- Step 4 can be slower because of repeated ARIMA fits and LSTM training.
- Current benchmark names include “CEEMDAN-*” approximations generated from VMD outputs + noise (as coded in `step4_models.py`).
- If you regenerate datasets yourself, ensure `master_weekly_prices.csv` keeps the expected columns:
  - `silver, gold, brent, dxy, sp500, vix, trends_raw`

---

## 7) Quick command block

```bash
python step1_vmd_decompose.py
python step2_entropy.py
python -c "import pandas as pd, numpy as np; n=int(len(pd.read_csv('master_weekly_prices.csv'))*0.8); np.save('n_train.npy', np.array([n])); print('Saved n_train.npy with n_train =', n)"
python step3_lasso.py
python step4_models.py
python step5_dmtest.py
python step6_trading.py
```
