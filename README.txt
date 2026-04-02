Silver Price Forecasting Pipeline - Indian Market
VMD + Entropy + LASSO + ARIMA/LSTM + Trading

This repository implements a weekly silver-price forecasting pipeline for the Indian market over a 25-year window (2000-2026), adapting the methodology of Liu et al. (2025).

Pipeline summary:
- Decomposition: VMD (Variational Mode Decomposition)
- Complexity split: Approximate Entropy (low vs high complexity IMFs)
- Feature selection: LASSO per IMF
- Forecasting: Hybrid ARIMA/LSTM plus benchmark models
- Evaluation: Error metrics, Diebold-Mariano test, and trading simulation

Project layout:
- download_financial_data.py - downloads silver, gold, brent, nifty50, and usdinr into financial_data/pre_processed/
- fetch_vix.py - downloads VIX from Yahoo Finance
- fetch_trends.py - downloads 12 India-specific silver Google Trends keywords
- build_master.py - merges all sources into financial_data/processed/master_weekly_prices.csv
- step1_vmd_decompose.py - performs VMD and creates imfs.npy, silver_weekly.csv, n_train.npy
- step2_entropy.py - computes IMF complexity labels
- step3_lasso.py - selects features for each IMF
- step4_models.py - trains forecasting models and saves predictions.pkl
- step5_dmtest.py - runs Diebold-Mariano testing
- step6_trading.py - runs the trading simulation

Data folders:
- financial_data/pre_processed/ - raw source files
- financial_data/processed/ - derived datasets and downstream inputs

Main processed dataset:
- financial_data/processed/master_weekly_prices.csv

Core outputs by stage:
- Step 1: fig4_silver_price_split.png, fig7_imf_decomposition.png, table5_imf_statistics.csv, imfs.npy, silver_weekly.csv
- Step 2: fig8_approximate_entropy.png, imf_complexity.csv
- Step 3: table6_lasso_features.csv, lasso_selected_features.pkl
- Step 4: table7_single_model_errors.csv, table8_decomp_model_errors.csv, fig9_single_model_forecasts.png, fig10_error_barplots.png, predictions.pkl
- Step 5: table9_dm_test.csv
- Step 6: table10_interval_errors.csv, table11_decomp_trading.csv, table12_single_trading.csv, fig11_interval_forecasts.png, fig12_trading_strategy_illustration.png, fig13_trading_evaluation.png

Quick setup:
1. Install dependencies.
2. Run python download_financial_data.py
3. Run python fetch_vix.py
4. Run python fetch_trends.py
5. Run python build_master.py
6. Run the modelling steps in order from step1_vmd_decompose.py through step6_trading.py

Notes:
- Run scripts from the repository root.
- Step 4 can take longer because of repeated ARIMA fits and LSTM training.
- The project uses Indian-market substitutions such as Nifty 50 instead of S&P 500 and USD/INR instead of DXY.
