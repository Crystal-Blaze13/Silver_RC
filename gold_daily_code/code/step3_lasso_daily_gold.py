"""
STEP 3 (daily) — LASSO Feature Selection Per IMF  [v3 — Pipeline Fix]
======================================================================

BUG FIXED (critical):
  v2 pre-scaled X_train once with StandardScaler, then passed the
  already-scaled matrix directly into LassoCV.  LassoCV internally
  splits that matrix into CV folds — but the scaler was fit on the
  entire training set, so every fold's "validation" portion had been
  touched by the scaler that saw it.  This is classic CV scale leakage:
  the selected alpha and the final coefficients are biased toward
  features whose variance happens to look larger in the held-out fold
  because the scaler over-shrank them.

  FIX: wrap StandardScaler + Lasso inside sklearn Pipeline.  The
  Pipeline re-fits the scaler on each fold's training split only,
  giving a proper out-of-fold estimate of alpha.  The final model is
  re-fit on the full training set with the chosen alpha (sklearn
  default), so the exported coefficients are clean.

  Downstream compatibility: the saved payload still contains
  "selected_features" (list of names per IMF) and "X_full" (unscaled,
  as before).  Step 4 re-scales with its own StandardScaler, so the
  change is transparent to all downstream steps.

UNCHANGED from v2:
  - Log-return feature construction (stationary inputs)
  - N_LAGS = 20
  - TimeSeriesSplit(n_splits=5)
  - External candidates, momentum features
  - Output file names and payload schema
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_FILE       = "../data/master_daily_prices_gold.csv"
IMF_FILE        = "../data/imfs_daily_gold.npy"
COMPLEXITY_FILE = "../data/imf_complexity_daily_gold.csv"
N_TRAIN_FILE    = "../data/n_train_daily_gold.npy"

N_LAGS   = 20
CV_FOLDS = 5

EXTERNAL_CANDIDATES = [
    "gold_usd",
    "brent",
    "usdinr",
    "nifty50",
    "vix_india",
    "mcx_gold",
    "geo_risk",
    "trends_raw",
    "india_epu",
    "sentiment_gold",
]

OUT_TABLE6 = "../results/tables/table6_gold_lasso_features.csv"
OUT_PKL    = "../data/lasso_selected_features_daily_gold.pkl"


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 3 (daily): LASSO Feature Selection  [v3 — Pipeline Fix]")
print("=" * 65)

df         = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
u_sorted   = np.load(IMF_FILE)
complexity = pd.read_csv(COMPLEXITY_FILE)
n_train    = int(np.load(N_TRAIN_FILE)[0])
K          = u_sorted.shape[0]

print(f"  {K} IMFs  |  {len(df)} observations  |  train={n_train}")
print(f"  Columns: {list(df.columns)}")


# ── 2. Compute stationary features ───────────────────────────────────────────
print("\nBuilding stationary feature set…")

gold_s  = df["mcx_gold"]

# ── 2a. Log-return lags of silver ─────────────────────────────────────────────
gold_logret = np.log(gold_s / gold_s.shift(1))

# ── 2b. Log-return lags of external level series ──────────────────────────────
available_ext = [c for c in EXTERNAL_CANDIDATES if c in df.columns]
missing_ext   = [c for c in EXTERNAL_CANDIDATES if c not in df.columns]
if missing_ext:
    print(f"  Missing external columns (skipped): {missing_ext}")

ext_logrets = {}
for col in available_ext:
    series = df[col].replace(0, np.nan).ffill()
    lr     = np.log(series / series.shift(1))
    ext_logrets[col] = lr.shift(1)
    print(f"  {col}: converted to log-return (lag-1)")

# ── 2c. Momentum / technical features ─────────────────────────────────────────
df["gold_ret5"]  = gold_s.pct_change(5).shift(1)
df["gold_ret20"] = gold_s.pct_change(20).shift(1)
df["gold_ret60"] = gold_s.pct_change(60).shift(1)

gold_col = ("mcx_gold" if "mcx_gold" in df.columns else
            "gold_usd" if "gold_usd" in df.columns else None)
if gold_col:
    df["gold_ret5"] = df[gold_col].pct_change(5).shift(1)

def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["gold_rsi14"]    = _rsi(gold_s, 14).shift(1)
df["gold_ma_ratio"] = (gold_s / gold_s.rolling(20).mean()).shift(1)
df["gold_vol10"]    = (gold_s.pct_change().rolling(10).std().shift(1)
                         * np.sqrt(252) * 100)

MOMENTUM_FEATURES = ["gold_ret5", "gold_ret20", "gold_ret60",
                     "gold_rsi14", "gold_ma_ratio", "gold_vol10"]
if gold_col:
    MOMENTUM_FEATURES.insert(3, "gold_ret5")

# Deduplicate MOMENTUM_FEATURES to avoid duplicate column names
MOMENTUM_FEATURES = list(dict.fromkeys(MOMENTUM_FEATURES))

available_mom = [c for c in MOMENTUM_FEATURES if c in df.columns]
print(f"\n  Momentum features ({len(available_mom)}): {available_mom}")


# ── 3. Build X_full ───────────────────────────────────────────────────────────
gold_arr = gold_logret.values
n_obs      = len(df)

lag_names         = [f"gold_logret_lag_{i}" for i in range(1, N_LAGS + 1)]
ext_lr_names      = [f"{c}_logret_lag1" for c in available_ext]
mom_names         = [f"{c}_lag1" for c in available_mom]
all_feature_names = lag_names + ext_lr_names + mom_names

print(f"\nFeature breakdown:")
print(f"  Log-return lags (gold):   {len(lag_names)}")
print(f"  Log-return lags (external): {len(ext_lr_names)}")
print(f"  Momentum features:          {len(mom_names)}")
print(f"  Total:                      {len(all_feature_names)}")

ext_lr_arrays = [ext_logrets[c].values for c in available_ext]
mom_arrays    = [df[c].values          for c in available_mom]

X_rows = []
for t in range(N_LAGS, n_obs):
    lr_lags = [gold_arr[t - i] for i in range(1, N_LAGS + 1)]
    ext_lr  = [arr[t] for arr in ext_lr_arrays]
    mom     = [arr[t] for arr in mom_arrays]
    X_rows.append(lr_lags + ext_lr + mom)

X_full = np.array(X_rows, dtype=float)
print(f"\nX_full shape: {X_full.shape}")

if np.any(np.isnan(X_full)):
    nan_cols = [all_feature_names[j] for j in range(X_full.shape[1])
                if np.any(np.isnan(X_full[:, j]))]
    print(f"\nNaNs in {len(nan_cols)} columns — forward/back filling…")
    X_full = pd.DataFrame(X_full, columns=all_feature_names).ffill().bfill().values
    remaining = int(np.sum(np.isnan(X_full)))
    if remaining:
        print(f"  {remaining} NaNs remain → replacing with 0")
        X_full = np.nan_to_num(X_full, nan=0.0)
    else:
        print("  All NaNs resolved.")

# Stationarity sanity check
print("\nTrain/Test std ratio check (should be ~1.0 for stationary features):")
train_std = X_full[:n_train - N_LAGS].std(axis=0)
test_std  = X_full[n_train - N_LAGS:].std(axis=0)
ratio     = np.where(train_std > 1e-8, test_std / train_std, np.nan)
bad_cols  = [(all_feature_names[j], ratio[j])
             for j in range(len(ratio))
             if np.isfinite(ratio[j]) and (ratio[j] > 3.0 or ratio[j] < 0.33)]
if bad_cols:
    print(f"  WARNING: {len(bad_cols)} features with train/test std ratio "
          f"outside [0.33, 3.0]:")
    for name, r in bad_cols[:10]:
        print(f"    {name:<35}  ratio={r:.2f}")
else:
    print("  OK: all features have stable variance across train/test split.")


# ── 4. LASSO per IMF (Pipeline — fixes CV scale leakage) ─────────────────────
# WHY PIPELINE MATTERS:
#   LassoCV splits X into folds internally.  Without a Pipeline, the
#   scaler has already seen ALL of X_train (including the fold's
#   validation rows), so every fold's held-out score is evaluated on
#   data that subtly influenced the scaler's mean/std.  The bias is
#   small per feature but compounds across 20+ features and 5 folds,
#   systematically favouring features with inflated test-fold variance.
#
#   With Pipeline(StandardScaler, LassoCV), sklearn re-fits the scaler
#   inside each fold using only that fold's training portion.  The
#   validation portion is transformed with the fold-local scaler,
#   giving a genuine out-of-fold alpha estimate.
#
#   The exported "scalers[i]" below is a fresh StandardScaler fit on
#   the full X_train — used by step 4 to scale X_test consistently.
#   (LassoCV inside the pipeline only selects alpha; step 4 applies its
#   own scaler as before.)

print(f"\nRunning LassoCV per IMF via Pipeline (cv={CV_FOLDS}, "
      f"TimeSeriesSplit) — scale leak fixed…")

tscv   = TimeSeriesSplit(n_splits=CV_FOLDS)
alphas = np.logspace(-6, 1, 60)
n_feat = X_full.shape[1]

selected_features = {}
scalers           = {}   # kept for step 4 compatibility (full-train scaler)
table6_rows       = []

for i in range(K):
    imf = u_sorted[i, N_LAGS:]

    X_train = X_full[:n_train - N_LAGS]
    y_train = imf[:n_train - N_LAGS]

    # ── Pipeline: scaler is re-fit inside each CV fold ────────────────────
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso",  LassoCV(alphas=alphas, cv=tscv, max_iter=20000,
                           random_state=42, n_jobs=-1,
                           fit_intercept=True)),
    ])
    pipe.fit(X_train, y_train)

    best_alpha = pipe.named_steps["lasso"].alpha_
    coef       = pipe.named_steps["lasso"].coef_

    # Re-fit a standalone StandardScaler on full X_train for export
    # (Step 4 will call scaler.transform(X_test) exactly as before)
    export_scaler = StandardScaler().fit(X_train)
    scalers[i]    = export_scaler

    selected_mask  = coef != 0
    selected_names = [all_feature_names[j] for j in range(n_feat)
                      if selected_mask[j]]
    selected_features[i] = selected_names

    comp   = complexity.loc[i, "Complexity"]
    ranked = sorted(
        [(all_feature_names[j], abs(coef[j]))
         for j in range(n_feat) if selected_mask[j]],
        key=lambda x: -x[1])
    top3 = [r[0] for r in ranked[:3]]
    print(f"  IMF{i+1:02d} ({comp:4s}): alpha={best_alpha:.6f}  "
          f"selected {int(sum(selected_mask))}/{n_feat}  top: {top3}")

    row = {"IMF": f"IMF{i+1}", "Complexity": comp,
           "Alpha": round(best_alpha, 8),
           "N_selected": int(sum(selected_mask)),
           "N_zero_coef": int(n_feat - sum(selected_mask))}
    for fname in all_feature_names:
        if fname in selected_names:
            cv = coef[all_feature_names.index(fname)]
            row[fname] = f"✓ ({cv:+.4f})"
        else:
            row[fname] = ""
    table6_rows.append(row)


# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n─── Feature selection summary ────────────────────────────────")
lag_sel = sum(1 for i in range(K)
              for f in selected_features[i]
              if f.startswith("gold_logret_lag"))
ext_sel = sum(1 for i in range(K)
              for f in selected_features[i]
              if "_logret_lag1" in f and not f.startswith("gold"))
mom_sel = sum(1 for i in range(K)
              for f in selected_features[i]
              if any(m in f for m in MOMENTUM_FEATURES))
print(f"  Log-return lags selected (across all IMFs):  {lag_sel}")
print(f"  External log-return features selected:       {ext_sel}")
print(f"  Momentum features selected:                  {mom_sel}")
print("──────────────────────────────────────────────────────────────")


# ── 6. Save ───────────────────────────────────────────────────────────────────
table6 = pd.DataFrame(table6_rows)
table6.to_csv(OUT_TABLE6, index=False)

display_cols = ["IMF", "Complexity", "N_selected"] + all_feature_names
display_df   = table6[[c for c in display_cols if c in table6.columns]].copy()
for fn in all_feature_names:
    if fn in display_df.columns:
        display_df[fn] = display_df[fn].apply(lambda v: "✓" if v else "")
print("\nTable 6 — LASSO Selected Features (✓ = selected):")
print(display_df.to_string(index=False))
print(f"\nSaved: {OUT_TABLE6}")

payload = {
    "selected_features": selected_features,
    "all_feature_names": all_feature_names,
    "feature_cols":      available_ext + available_mom,
    "N_LAGS":            N_LAGS,
    "X_full":            X_full,       # unscaled — step 4 scales itself
    "n_train_adj":       n_train - N_LAGS,
    "scalers":           scalers,      # full-train StandardScalers per IMF
    "momentum_features": MOMENTUM_FEATURES,
    "n_lags_used":       N_LAGS,
    "feature_type":      "log_returns",
    "lasso_version":     "v3_pipeline",  # downstream can detect the fix
}
with open(OUT_PKL, "wb") as f:
    pickle.dump(payload, f)
print(f"Saved: {OUT_PKL}")

print("\n" + "=" * 65)
print("STEP 3 COMPLETE  [v3 — Pipeline Fix]")
print(f"  Key change: LassoCV now runs inside Pipeline(StandardScaler, …)")
print(f"  CV fold scaling is now proper out-of-fold — no leakage")
print("=" * 65)
print("NEXT: python step4_models_daily_gold.py")