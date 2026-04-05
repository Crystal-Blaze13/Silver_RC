"""
DIAGNOSTIC — run this before step4 to understand why ARIMA and LSTM are failing.
Place in your daily/ directory and run:
  python diagnose_step4.py
"""
import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

IMF_FILE        = "../processed/imfs_daily.npy"
COMPLEXITY_FILE = "../processed/imf_complexity_daily.csv"
SILVER_FILE     = "../processed/silver_daily.csv"
N_TRAIN_FILE    = "../processed/n_train_daily.npy"
LASSO_FILE      = "../processed/lasso_selected_features_daily.pkl"
import pickle

u_sorted   = np.load(IMF_FILE)
complexity = pd.read_csv(COMPLEXITY_FILE)
n_train    = int(np.load(N_TRAIN_FILE)[0])
K          = u_sorted.shape[0]

silver_df    = pd.read_csv(SILVER_FILE, index_col=0, parse_dates=True)
silver_price = silver_df.iloc[:, 0].values

with open(LASSO_FILE, "rb") as f:
    lasso_data = pickle.load(f)
N_LAGS      = lasso_data["N_LAGS"]
n_train_adj = lasso_data["n_train_adj"]

print("=" * 70)
print("DIAGNOSTIC: IMF properties and ARIMA fit check")
print("=" * 70)
print(f"\nSilver price range: {silver_price.min():,.0f} – {silver_price.max():,.0f} INR/kg")
print(f"Silver price mean:  {silver_price.mean():,.0f}")
print(f"Silver price std:   {silver_price.std():,.0f}")
print(f"\nN_LAGS={N_LAGS}  n_train={n_train}  n_train_adj={n_train_adj}  K={K}")

print("\n── IMF summary ──────────────────────────────────────────────────────")
for i in range(K):
    imf  = u_sorted[i, N_LAGS:]
    comp = complexity.loc[i, "Complexity"]
    y_tr = imf[:n_train_adj]
    adf_p = adfuller(y_tr, autolag="AIC")[1]
    print(f"  IMF{i+1:02d} ({comp:4s})  mean={y_tr.mean():+10.2f}  "
          f"std={y_tr.std():8.2f}  min={y_tr.min():+10.2f}  "
          f"max={y_tr.max():+10.2f}  ADF_p={adf_p:.4f}")

print("\n── ARIMA fit check on Low-complexity IMFs ───────────────────────────")
WF_BURN_IN = 120
for i in range(K):
    imf  = u_sorted[i, N_LAGS:]
    comp = complexity.loc[i, "Complexity"]
    if comp != "Low":
        continue
    y_tr = imf[:n_train_adj]
    print(f"\n  IMF{i+1:02d} (Low):")

    # Try every order and report which ones succeed
    successes = []
    failures  = []
    for p, d, q in itertools.product(range(5), range(2), range(3)):
        try:
            res = ARIMA(y_tr, order=(p, d, q)).fit()
            successes.append(((p, d, q), round(res.aic, 2)))
        except Exception as e:
            failures.append(((p, d, q), str(e)[:60]))

    if successes:
        best = min(successes, key=lambda x: x[1])
        print(f"    ✓ {len(successes)}/30 orders fit OK.  Best: {best[0]}  AIC={best[1]}")
        # show top 5
        top5 = sorted(successes, key=lambda x: x[1])[:5]
        for o, aic in top5:
            print(f"      {o}  AIC={aic}")
    else:
        print(f"    ✗ ALL 30 orders FAILED — ARIMA cannot fit this IMF at all!")

    if failures:
        print(f"    ✗ {len(failures)} failures.  First few error types:")
        seen_errs = set()
        for o, err in failures[:10]:
            short = err.split(".")[0][:50]
            if short not in seen_errs:
                print(f"      {o}: {short}")
                seen_errs.add(short)

print("\n── Reconstruction check ─────────────────────────────────────────────")
recon  = u_sorted.sum(axis=0)
target = silver_price[N_LAGS:N_LAGS + len(recon)]
if len(target) == len(recon):
    err = np.abs(recon - target)
    print(f"  Max recon error: {err.max():,.4f}")
    print(f"  Mean recon error: {err.mean():,.4f}")
    # Check if any IMF has level in the thousands (not differences)
    for i in range(K):
        imf = u_sorted[i, N_LAGS:]
        if np.abs(imf).max() > 1000:
            print(f"  WARNING: IMF{i+1} has values up to {np.abs(imf).max():,.0f} "
                  f"— may be in price LEVELS not components!")
else:
    print(f"  Shape mismatch: recon={len(recon)}, target={len(target)}")

print("\n── LSTM cumsum drift check ───────────────────────────────────────────")
print("  Simulating what happens when predicted diffs have a small bias:")
for bias in [0.01, 0.1, 1.0]:
    n_steps = len(silver_price) - n_train
    drift = np.abs(bias * n_steps)
    print(f"  diff_bias={bias:+.3f}/step × {n_steps} steps = "
          f"cumulative drift={drift:,.1f} INR/kg")

print("\n── n_train_adj vs n_train check ─────────────────────────────────────")
print(f"  n_train={n_train}  n_train_adj={n_train_adj}  N_LAGS={N_LAGS}")
print(f"  n_train - N_LAGS = {n_train - N_LAGS}  (should equal n_train_adj)")
if n_train - N_LAGS != n_train_adj:
    print(f"  *** MISMATCH! This will cause IMF slice misalignment ***")
else:
    print(f"  OK — alignment matches")

print("\nDIAGNOSTIC COMPLETE")
