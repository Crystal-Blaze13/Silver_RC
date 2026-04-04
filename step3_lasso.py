"""
STEP 3 — LASSO Feature Selection Per IMF
=========================================
Produces: Table 6
Input:    imfs.npy, imf_complexity.csv, master_weekly_prices.csv, n_train.npy
Outputs:  table6_lasso_features.csv
          lasso_selected_features.pkl  (used by steps 4–6)

Target:   MCX Silver (INR/kg) — Indian market benchmark.

Feature candidates (14 total):
  Lags 1–5 of mcx_silver (autoregressive)
  External (lagged 1 period to prevent look-ahead bias):
    gold_usd   — CME Gold futures (USD/oz)  — global safe-haven / commodity
    brent      — ICE Brent crude (USD/bbl)  — energy cost / USD demand
    usdinr     — USD/INR                    — critical for Indian import-priced silver
    nifty50    — India equity index          — domestic risk appetite
    vix_india  — India VIX (INVIXN)         — domestic fear gauge
    mcx_gold   — MCX Gold (INR/10 g)        — Indian gold-silver link
    geo_risk   — GPRXGPRD geopolitical risk — import cost driver
    trends_raw — Google Trends (India silver keywords, 0–100)
    india_epu  — India EPU index (Baker, Bloom & Davis)

IMPROVEMENTS OVER ORIGINAL:
  - Validates all required columns exist before building X; missing external
    features are skipped with a warning rather than crashing.
  - LassoCV uses time-series-safe walk-forward (TimeSeriesSplit) instead of
    random k-fold so no future leakage in cross-validation.
  - Alpha grid is log-spaced and constrained to avoid degenerate (all-zero)
    solutions on smooth IMFs.
  - Adds 'N_zero_coef' column to Table 6 (features explicitly zeroed out).
  - Saves scaler per IMF so step 4 can re-use exact standardisation.
  - Prints feature importance ranking (|coef|) for each IMF.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────
DATA_FILE       = "financial_data/processed/master_weekly_prices.csv"
IMF_FILE        = "imfs.npy"
COMPLEXITY_FILE = "imf_complexity.csv"
N_TRAIN_FILE    = "financial_data/processed/n_train.npy"

N_LAGS   = 5    # paper uses up to 5 lagged values
CV_FOLDS = 5    # time-series cross-validation folds for LassoCV

# All potential external features (lagged by 1 period)
EXTERNAL_CANDIDATES = [
    'gold_usd',   # CME Gold (USD/oz)
    'brent',      # ICE Brent crude (USD/bbl)
    'usdinr',     # USD/INR exchange rate
    'nifty50',    # NSE Nifty 50 index
    'vix_india',  # India VIX
    'mcx_gold',   # MCX Gold (INR/10 g)
    'geo_risk',   # Geopolitical Risk index (GPRXGPRD)
    'trends_raw', # Google Trends composite (0–100)
    'india_epu',  # India EPU index
]

OUT_TABLE6 = "table6_lasso_features.csv"
OUT_PKL    = "lasso_selected_features.pkl"

# ── 1. Load data ───────────────────────────────────────────────
print("=" * 60)
print("STEP 3: LASSO Feature Selection")
print("=" * 60)

df         = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
u_sorted   = np.load(IMF_FILE)
complexity = pd.read_csv(COMPLEXITY_FILE)
n_train    = int(np.load(N_TRAIN_FILE)[0])
K          = u_sorted.shape[0]

print(f"Loaded {K} IMFs, {len(df)} weekly observations")
print(f"Training on first {n_train} observations")
print(f"Columns in master file: {list(df.columns)}")

# ── 2. Validate & select external features ────────────────────
available_ext = [c for c in EXTERNAL_CANDIDATES if c in df.columns]
missing_ext   = [c for c in EXTERNAL_CANDIDATES if c not in df.columns]

if missing_ext:
    print(f"\nWARNING: {len(missing_ext)} external features not found, "
          f"skipping: {missing_ext}")
if not available_ext:
    print("WARNING: No external features found! Using only AR lags.")

print(f"\nUsing {len(available_ext)} external features: {available_ext}")

# ── 3. Build feature matrix ───────────────────────────────────
silver    = df['mcx_silver'].values
externals = df[available_ext].values if available_ext else None
n_obs     = len(df)

lag_names     = [f"mcx_silver_lag_{i}" for i in range(1, N_LAGS + 1)]
ext_lag_names = [f"{c}_lag1" for c in available_ext]
all_feature_names = lag_names + ext_lag_names

X_rows = []
for t in range(N_LAGS, n_obs):
    lags = [silver[t - i] for i in range(1, N_LAGS + 1)]
    ext  = list(externals[t - 1]) if externals is not None else []
    X_rows.append(lags + ext)

X_full  = np.array(X_rows, dtype=float)
n_feat  = X_full.shape[1]

print(f"Feature matrix shape: {X_full.shape}")
print(f"Features: {all_feature_names}")

# Guard against NaNs
if np.any(np.isnan(X_full)):
    nan_cols = [all_feature_names[j]
                for j in range(n_feat)
                if np.any(np.isnan(X_full[:, j]))]
    print(f"WARNING: NaNs detected in: {nan_cols}. Forward-filling…")
    X_full = pd.DataFrame(X_full, columns=all_feature_names).ffill().bfill().values

# ── 4. Run LASSO per IMF (training portion only) ───────────────
print(f"\nRunning LassoCV (TimeSeriesSplit, cv={CV_FOLDS}) for each IMF…")

tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

# Log-spaced alpha grid that covers weak (≈0) to strong (≈1) regularisation
alphas = np.logspace(-6, 1, 60)

selected_features = {}   # IMF index → list of selected feature names
scalers           = {}   # IMF index → fitted StandardScaler
table6_rows       = []

for i in range(K):
    imf = u_sorted[i, N_LAGS:]   # align with feature matrix

    X_train = X_full[:n_train - N_LAGS]
    y_train = imf[:n_train - N_LAGS]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    scalers[i] = scaler

    lasso = LassoCV(
        alphas=alphas,
        cv=tscv,
        max_iter=20000,
        random_state=42,
        n_jobs=-1,
    )
    lasso.fit(X_scaled, y_train)

    coef          = lasso.coef_
    selected_mask = coef != 0
    selected_names = [all_feature_names[j]
                      for j in range(n_feat) if selected_mask[j]]
    selected_features[i] = selected_names

    complexity_label = complexity.loc[i, 'Complexity']

    # Feature importance (|coef| ranking)
    ranked = sorted(
        [(all_feature_names[j], abs(coef[j]))
         for j in range(n_feat) if selected_mask[j]],
        key=lambda x: -x[1]
    )

    print(f"  IMF{i+1:02d} ({complexity_label:4s}): "
          f"alpha={lasso.alpha_:.6f}  "
          f"selected {sum(selected_mask)}/{n_feat}  "
          f"top: {[r[0] for r in ranked[:3]]}")

    row = {
        "IMF":        f"IMF{i+1}",
        "Complexity": complexity_label,
        "Alpha":      round(lasso.alpha_, 8),
        "N_selected": int(sum(selected_mask)),
        "N_zero_coef": int(n_feat - sum(selected_mask)),
    }
    for fname in all_feature_names:
        if fname in selected_names:
            coef_val = coef[all_feature_names.index(fname)]
            row[fname] = f"✓ ({coef_val:+.4f})"
        else:
            row[fname] = ""
    table6_rows.append(row)

# ── 5. Save Table 6 ───────────────────────────────────────────
table6 = pd.DataFrame(table6_rows)
table6.to_csv(OUT_TABLE6, index=False)

# Print a readable tick-only version
display_cols = ["IMF", "Complexity", "N_selected"] + all_feature_names
display_df = table6[display_cols].copy()
for fn in all_feature_names:
    display_df[fn] = display_df[fn].apply(lambda v: "✓" if v else "")
print("\nTable 6 — LASSO Selected Features (✓ = selected):")
print(display_df.to_string(index=False))
print(f"\nSaved: {OUT_TABLE6}")

# ── 6. Save artefacts for later steps ─────────────────────────
payload = {
    "selected_features": selected_features,
    "all_feature_names": all_feature_names,
    "feature_cols":      available_ext,
    "N_LAGS":            N_LAGS,
    "X_full":            X_full,
    "n_train_adj":       n_train - N_LAGS,
    "scalers":           scalers,           # ← NEW: per-IMF fitted scalers
}
with open(OUT_PKL, "wb") as f:
    pickle.dump(payload, f)
print(f"Saved: {OUT_PKL}")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print(f"  {OUT_TABLE6}      ← Table 6")
print(f"  {OUT_PKL}  ← needed by steps 4–6")
print("=" * 60)
print("NEXT: python3 step4_models.py")