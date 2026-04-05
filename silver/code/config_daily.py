"""
config_daily.py — Configuration for Daily MCX Silver Forecasting Pipeline
=========================================================================
Mirrors config.py but adapted for daily (business-day) frequency from 2015.

Key differences vs weekly pipeline:
  - START_DATE: 2015-01-01 (GDELT data starts ~2015-02)
  - Frequency: 'B' (business days) instead of 'W-SUN'
  - LSTM_SEQ_LEN: 20 business days (~4 calendar weeks)
  - WF_BURN_IN: 120 business days (~6 months)
  - Sentiment: real GDELT daily avg_tone from three pre-downloaded CSVs
"""

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_FILE    = "../data/master_daily_prices.csv"
SILVER_FILE  = "../data/silver_daily.csv"
N_TRAIN_FILE = "../data/n_train_daily.npy"

SILVER_COL  = "mcx_silver"
START_DATE  = "2015-01-01"
TEST_START  = "2024-03-01"   # first date in test set

# GDELT CSV files (relative to silver/code/ folder)
GDELT_FILES = [
    "../../common_data/bquxjob_178470f6_19d57f8b1bb.csv",  # 2015-02-18 → 2018-12-31
    "../../common_data/bquxjob_12d5030e_19d57fb8d7d.csv",  # 2019-01-01 → 2022-12-31
    "../../common_data/bquxjob_49489022_19d57fc378f.csv",  # 2023-01-01 → 2026-03-31
]

# ── Step 1: VMD decomposition ─────────────────────────────────────────────────
VMD_ALPHA = 2000
VMD_TAU   = 0
VMD_DC    = 0
VMD_INIT  = 1
VMD_TOL   = 1e-7
VMD_K_MIN = 4
VMD_K_MAX = 12
VMD_RESIDUAL_THRESHOLD_PCT = 0.5

# ── Step 2: Entropy ───────────────────────────────────────────────────────────
AE_M      = 2
AE_R_COEF = 0.2
SAMPEN_M  = 2
SAMPEN_R  = 0.2

# ── Step 3: LASSO ────────────────────────────────────────────────────────────
N_LAGS   = 5        # 5 daily lags (~1 trading week)
CV_FOLDS = 5
EXTERNAL_CANDIDATES = [
    "gold_usd",
    "brent",
    "usdinr",
    "nifty50",
    "vix_india",
    "mcx_gold",
    "geo_risk",
    "trends_raw",   # Google Trends (weekly → daily ffill)
    "india_epu",    # EPU (monthly → daily ffill)
    "sentiment_silver",  # GDELT daily avg_tone for silver
]

# ── Step 4: Models ────────────────────────────────────────────────────────────
LSTM_EPOCHS  = 150
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 1
LSTM_LR      = 0.001
LSTM_SEQ_LEN = 20    # 20 business days ≈ 4 calendar weeks
VAL_FRAC     = 0.10
ELM_HIDDEN   = 50
WF_BURN_IN   = 120   # 120 business days ≈ 6 months
USE_CEEMDAN  = True

# ── Step 6: Trading ───────────────────────────────────────────────────────────
TRANSACTION_COST = 0.0005
INITIAL_CAPITAL  = 1.0
CONFORMAL_ALPHA  = 0.10
CAL_FRAC         = 0.20
IMLP_HIDDEN      = (64, 32)
IMLP_EPOCHS      = 500
IMLP_LR          = 5e-4
IMLP_PATIENCE    = 30
IMLP_SEED        = 42
VOL_WINDOW       = 20   # 20-day rolling vol (1 month)

# ── Output files — figures ────────────────────────────────────────────────────
FIG4   = "../results/figures/fig4_silver_price_split.png"
FIG6   = "../results/figures/fig6_sentiment.png"
FIG7   = "../results/figures/fig7_imf_decomposition.png"
FIG8   = "../results/figures/fig8_approximate_entropy.png"
FIG9   = "../results/figures/fig9_single_model_forecasts.png"
FIG10  = "../results/figures/fig10_error_barplots.png"
FIG11  = "../results/figures/fig11_interval_forecasts.png"
FIG12  = "../results/figures/fig12_trading_strategy_illustration.png"
FIG13  = "../results/figures/fig13_trading_evaluation.png"
FIG13B = "../results/figures/fig13b_equity_curves.png"
FIG_DM = "../results/figures/fig_dm_heatmap.png"

# ── Output files — tables ─────────────────────────────────────────────────────
TABLE5  = "../results/tables/table5_imf_statistics.csv"
TABLE6  = "../results/tables/table6_lasso_features.csv"
TABLE7  = "../results/tables/table7_single_model_errors.csv"
TABLE8  = "../results/tables/table8_decomp_model_errors.csv"
TABLE9  = "../results/tables/table9_dm_test.csv"
TABLE10 = "../results/tables/table10_interval_errors.csv"
TABLE11 = "../results/tables/table11_decomp_trading.csv"
TABLE12 = "../results/tables/table12_single_trading.csv"

# ── Output files — intermediates ──────────────────────────────────────────────
IMFS_FILE       = "../data/imfs_daily.npy"
COMPLEXITY_FILE = "../data/imf_complexity_daily.csv"
LASSO_PKL       = "../data/lasso_selected_features_daily.pkl"
PREDICTIONS_PKL = "../data/predictions_daily.pkl"

# ── Figure style ──────────────────────────────────────────────────────────────
FIG_DPI   = 300
FIG_STYLE = "seaborn-v0_8-whitegrid"

PALETTE = {
    "train":    "#27ae60",
    "test":     "#e74c3c",
    "proposed": "#2c3e50",
    "arima":    "#e67e22",
    "lstm":     "#8e44ad",
    "vmd_arima":"#16a085",
    "low_comp": "#3498db",
    "high_comp":"#e74c3c",
    "interval": "#95a5a6",
    "neutral":  "#bdc3c7",
}
