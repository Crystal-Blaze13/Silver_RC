"""
config.py — Central configuration for MCX Silver Forecasting Pipeline
======================================================================
All path, hyperparameter, and output settings live here.
Each step file imports from this module so edits propagate everywhere.

Paper replicated: Liu et al. (2025) "From forecasting to trading:
A multimodal-data-driven approach to reversing carbon market losses"
Energy Economics 144, 108350.

Adaptation: MCX Silver (INR/kg), Indian market, weekly frequency, 2008–2026.
"""

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_FILE   = "../processed/master_weekly_prices.csv"
SILVER_FILE = "../processed/silver_weekly.csv"
N_TRAIN_FILE = "../processed/n_train.npy"

SILVER_COL  = "mcx_silver"
TEST_START  = "2024-03-01"   # first date in test set

# ── Step 1: VMD decomposition ─────────────────────────────────────────────────
VMD_ALPHA = 2000   # bandwidth penalty (paper value)
VMD_TAU   = 0      # noise tolerance
VMD_DC    = 0      # no DC mode
VMD_INIT  = 1      # omega initialised uniformly
VMD_TOL   = 1e-7   # convergence tolerance
VMD_K_MIN = 4      # minimum IMFs allowed
VMD_K_MAX = 12     # maximum IMFs allowed
VMD_RESIDUAL_THRESHOLD_PCT = 0.5   # append residual as extra IMF above this %

# ── Step 2: Entropy ───────────────────────────────────────────────────────────
AE_M      = 2      # ApEn embedding dimension (standard)
AE_R_COEF = 0.2    # tolerance = 0.2 × std
SAMPEN_M  = 2
SAMPEN_R  = 0.2

# ── Step 3: LASSO ────────────────────────────────────────────────────────────
N_LAGS   = 5       # autoregressive lags (paper: up to 5)
CV_FOLDS = 5       # TimeSeriesSplit folds for LassoCV
EXTERNAL_CANDIDATES = [
    "gold_usd",    # CME Gold futures (USD/oz)
    "brent",       # ICE Brent crude (USD/bbl)
    "usdinr",      # USD/INR exchange rate
    "nifty50",     # NSE Nifty 50 index
    "vix_india",   # India VIX
    "mcx_gold",    # MCX Gold (INR/10g)
    "geo_risk",    # Geopolitical Risk index (GPRXGPRD)
    "trends_raw",  # Google Trends composite (0–100)
    "india_epu",   # India EPU index
]

# ── Step 4: Models ────────────────────────────────────────────────────────────
LSTM_EPOCHS  = 150
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 1
LSTM_LR      = 0.001
LSTM_SEQ_LEN = 4
VAL_FRAC     = 0.10    # fraction of train used for early-stopping validation
ELM_HIDDEN   = 50
WF_BURN_IN   = 24      # walk-forward burn-in weeks
USE_CEEMDAN  = True    # set False to skip CEEMDAN benchmarks (slow)

# ── Step 6: Trading ───────────────────────────────────────────────────────────
TRANSACTION_COST = 0.0005    # 0.05% per leg (paper value)
INITIAL_CAPITAL  = 1.0
CONFORMAL_ALPHA  = 0.10      # → 80% prediction interval
CAL_FRAC         = 0.20      # calibration split from training tail
IMLP_HIDDEN      = (64, 32)
IMLP_EPOCHS      = 500
IMLP_LR          = 5e-4
IMLP_PATIENCE    = 30
IMLP_SEED        = 42
VOL_WINDOW       = 10        # rolling vol window for scaled conformity

# ── Output files — figures ────────────────────────────────────────────────────
FIG4   = "../results/figures/fig4_silver_price_split.png"
FIG7   = "../results/figures/fig7_imf_decomposition.png"
FIG8   = "../results/figures/fig8_approximate_entropy.png"
FIG8B  = "../results/figures/fig8b_imf_correlation.png"
FIG9   = "../results/figures/fig9_single_model_forecasts.png"
FIG10  = "../results/figures/fig10_error_barplots.png"
FIG11  = "../results/figures/fig11_interval_forecasts.png"
FIG12  = "../results/figures/fig12_trading_strategy_illustration.png"
FIG13  = "../results/figures/fig13_trading_evaluation.png"
FIG13B = "../results/figures/fig13b_equity_curves.png"
FIG_DM = "../results/figures/fig_dm_heatmap.png"
FIG_STRESS = "../results/figures/fig_stress_test_comparison.png"

# ── Output files — tables ─────────────────────────────────────────────────────
TABLE5  = "../results/tables/table5_imf_statistics.csv"
TABLE6  = "../results/tables/table6_lasso_features.csv"
TABLE7  = "../results/tables/table7_single_model_errors.csv"
TABLE8  = "../results/tables/table8_decomp_model_errors.csv"
TABLE9  = "../results/tables/table9_dm_test.csv"
TABLE10 = "../results/tables/table10_interval_errors.csv"
TABLE11 = "../results/tables/table11_decomp_trading.csv"
TABLE12 = "../results/tables/table12_single_trading.csv"
TABLE13 = "../results/tables/table13_margin_sweep.csv"
TABLE14 = "../results/tables/table14_holding_periods.csv"
TABLE15 = "../results/tables/table15_final_comparison.csv"

# ── Output files — intermediates ──────────────────────────────────────────────
IMFS_FILE        = "../processed/imfs.npy"
COMPLEXITY_FILE  = "../processed/imf_complexity.csv"
LASSO_PKL        = "../processed/lasso_selected_features.pkl"
PREDICTIONS_PKL  = "../processed/predictions.pkl"
STRESS_PKL       = "../processed/predictions_synthetic_crash.pkl"
COMPARISON_JSON  = "../results/tables/comparison_actual_vs_synthetic.json"

# ── Figure style ──────────────────────────────────────────────────────────────
FIG_DPI     = 300          # publication quality
FIG_STYLE   = "seaborn-v0_8-whitegrid"

PALETTE = {
    "train":    "#27ae60",   # green
    "test":     "#e74c3c",   # red
    "proposed": "#2c3e50",   # dark slate
    "arima":    "#e67e22",   # orange
    "lstm":     "#8e44ad",   # purple
    "vmd_arima":"#16a085",   # teal
    "low_comp": "#3498db",   # blue
    "high_comp":"#e74c3c",   # red
    "interval": "#95a5a6",   # grey
    "neutral":  "#bdc3c7",
}
