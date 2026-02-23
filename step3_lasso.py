"""
STEP 3 — LASSO Feature Selection Per IMF
Produces: Table 6
Input:    imfs.npy, imf_complexity.csv, master_weekly_prices.csv, n_train.npy
Outputs:  table6_lasso_features.csv
          lasso_selected_features.pkl  (used by steps 4-6)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ── Settings ──────────────────────────────────────────────────
DATA_FILE       = "master_weekly_prices.csv"
IMF_FILE        = "imfs.npy"
COMPLEXITY_FILE = "imf_complexity.csv"
N_TRAIN_FILE    = "n_train.npy"
N_LAGS          = 5    # paper uses up to 5 lagged values
CV_FOLDS        = 5    # cross-validation folds for LassoCV

# ── 1. Load data ───────────────────────────────────────────────
print("=" * 55)
print("STEP 3: LASSO Feature Selection")
print("=" * 55)

df         = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
u_sorted   = np.load(IMF_FILE)
complexity = pd.read_csv(COMPLEXITY_FILE)
n_train    = int(np.load(N_TRAIN_FILE)[0])
K          = u_sorted.shape[0]

print(f"Loaded {K} IMFs, {len(df)} weekly observations")
print(f"Training on first {n_train} observations")

# ── 2. Build feature matrix ───────────────────────────────────
# Features: lagged silver price (1-5) + external variables
# External: gold, brent, dxy, sp500, vix, trends_raw
# This mirrors the paper's Table 3 feature set

feature_cols = ['gold', 'brent', 'dxy', 'sp500', 'vix', 'trends_raw']
silver       = df['silver'].values
externals    = df[feature_cols].values
n_obs        = len(df)

print(f"\nExternal features: {feature_cols}")
print(f"Lag features: silver_lag_1 to silver_lag_{N_LAGS}")

# Build lagged silver features
lag_names = [f"silver_lag_{i}" for i in range(1, N_LAGS + 1)]
all_feature_names = lag_names + feature_cols

# Build full feature matrix (starting from row N_LAGS to have all lags)
X_rows = []
for t in range(N_LAGS, n_obs):
    lags = [silver[t - i] for i in range(1, N_LAGS + 1)]
    ext  = list(externals[t])
    X_rows.append(lags + ext)

X_full = np.array(X_rows)   # shape: (n_obs - N_LAGS, n_features)
n_feat = X_full.shape[1]

print(f"Feature matrix shape: {X_full.shape}")

# ── 3. Run LASSO per IMF (on training portion only) ───────────
print("\nRunning LassoCV for each IMF...")

selected_features = {}   # IMF index → list of selected feature names
table6_rows       = []

for i in range(K):
    imf = u_sorted[i, N_LAGS:]   # align with feature matrix (drop first N_LAGS)

    # Training portion only
    X_train = X_full[:n_train - N_LAGS]
    y_train = imf[:n_train - N_LAGS]

    # Standardise features (important for LASSO)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # LassoCV: automatically finds best alpha via cross-validation
    lasso = LassoCV(cv=CV_FOLDS, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y_train)

    # Features with non-zero coefficients are selected
    selected_mask  = lasso.coef_ != 0
    selected_names = [all_feature_names[j] for j in range(n_feat) if selected_mask[j]]
    selected_features[i] = selected_names

    complexity_label = complexity.loc[i, 'Complexity']
    print(f"  IMF{i+1} ({complexity_label}): alpha={lasso.alpha_:.6f}, "
          f"selected {sum(selected_mask)}/{n_feat} features: {selected_names}")

    # Build row for Table 6
    row = {"IMF": f"IMF{i+1}", "Complexity": complexity_label,
           "Alpha": round(lasso.alpha_, 6),
           "N_selected": sum(selected_mask)}
    for fname in all_feature_names:
        row[fname] = "✓" if fname in selected_names else ""
    table6_rows.append(row)

# ── 4. Save Table 6 ───────────────────────────────────────────
table6 = pd.DataFrame(table6_rows)
table6.to_csv("table6_lasso_features.csv", index=False)

print("\nTable 6 — LASSO Selected Features:")
# Print a readable version
display_cols = ["IMF", "Complexity"] + all_feature_names
print(table6[display_cols].to_string(index=False))
print("\nSaved: table6_lasso_features.csv")

# ── 5. Save selected features for later steps ─────────────────
with open("lasso_selected_features.pkl", "wb") as f:
    pickle.dump({
        "selected_features": selected_features,
        "all_feature_names": all_feature_names,
        "feature_cols":      feature_cols,
        "N_LAGS":            N_LAGS,
        "X_full":            X_full,
        "n_train_adj":       n_train - N_LAGS,
    }, f)
print("Saved: lasso_selected_features.pkl")

print("\n" + "=" * 55)
print("STEP 3 COMPLETE")
print("  table6_lasso_features.csv  ← Table 6")
print("  lasso_selected_features.pkl ← needed by steps 4-6")
print("=" * 55)
print("NEXT: python3 step4_models.py")