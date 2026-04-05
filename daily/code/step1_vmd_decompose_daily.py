"""
STEP 1 (daily) — VMD Decomposition of MCX Silver Price
=======================================================
FIX applied (DA diagnostic):
  - After decomposition, prints IMF1 variance ratio with a clear
    WARNING if it falls below 0.65.  When the trend IMF is weak the
    whole decomposition → direction pipeline is compromised.
  - If the warning fires, re-run with ALPHA = 4000 (uncomment the
    override line near the top of Settings) or reduce K by 1.
  - All other logic is identical to the original.

Produces:
  fig4_silver_price_split.png
  fig7_imf_decomposition.png
  table5_imf_statistics.csv
  imfs_daily.npy          ← used by steps 2–6
  financial_data/processed/silver_daily.csv
  financial_data/processed/n_train_daily.npy

HOW TO RUN (from the daily/ directory):
  pip install vmdpy PyEMD pandas numpy matplotlib scipy
  python step1_vmd_decompose_daily.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from scipy import stats as _stats
from vmdpy import VMD

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_FILE   = "../processed/master_daily_prices.csv"
SILVER_COL  = "mcx_silver"
TEST_START  = "2024-03-01"

ALPHA = 2000        # ← if IMF1 variance ratio < 0.65, change to 4000
# ALPHA = 4000      # ← uncomment this line and comment the one above

TAU   = 0
DC    = 0
INIT  = 1
TOL   = 1e-7
VMD_RESIDUAL_THRESHOLD_PCT = 0.5

# Minimum acceptable variance ratio for IMF1 (trend component).
# Below this threshold the trend signal is too fragmented for reliable
# direction forecasting and a WARNING will be printed.
IMF1_VR_THRESHOLD = 0.65

OUT_FIG4   = "../results/figures/fig4_silver_price_split.png"
OUT_FIG7   = "../results/figures/fig7_imf_decomposition.png"
OUT_TAB5   = "../results/tables/table5_imf_statistics.csv"
OUT_IMFS   = "../processed/imfs_daily.npy"
OUT_SILVER = "../processed/silver_daily.csv"
OUT_NTRAIN = "../processed/n_train_daily.npy"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 (daily): VMD Decomposition")
print("=" * 60)

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
if SILVER_COL not in df.columns:
    raise ValueError(f"Column '{SILVER_COL}' not found. "
                     f"Available: {list(df.columns)}")

price = df[SILVER_COL].dropna()
print(f"Loaded {len(price)} daily observations")
print(f"Date range : {price.index[0].date()} → {price.index[-1].date()}")
print(f"Price range: {price.min():,.0f} – {price.max():,.0f} INR/kg")

os.makedirs(os.path.dirname(OUT_SILVER), exist_ok=True)
price.to_csv(OUT_SILVER, header=True)
print(f"Saved: {OUT_SILVER}")

# ── 2. Train / Test split ─────────────────────────────────────────────────────
split_date   = pd.Timestamp(TEST_START)
train_prices = price[price.index <  split_date]
test_prices  = price[price.index >= split_date]

n_train = len(train_prices)
n_test  = len(test_prices)

print(f"\nTrain: {n_train} days "
      f"({train_prices.index[0].date()} → {train_prices.index[-1].date()})")
print(f"Test : {n_test}  days "
      f"({test_prices.index[0].date()} → {test_prices.index[-1].date()})")

np.save(OUT_NTRAIN, np.array([n_train]))
print(f"Saved: {OUT_NTRAIN}")

# ── 2b. Structural break (Chow test) ─────────────────────────────────────────
def _chow_test(y1, y2):
    def _ols_rss(y):
        x = np.arange(len(y))
        X = np.column_stack([np.ones(len(x)), x])
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ b) ** 2)
        return rss, len(x)
    rss1, n1 = _ols_rss(y1.values)
    rss2, n2 = _ols_rss(y2.values)
    rss_pool, _ = _ols_rss(pd.concat([y1, y2]).values)
    k = 2
    F = ((rss_pool - rss1 - rss2) / k) / ((rss1 + rss2) / (n1 + n2 - 2 * k))
    p = 1 - _stats.f.cdf(F, dfn=k, dfd=n1 + n2 - 2 * k)
    return F, p

tr_slope, *_ = _stats.linregress(np.arange(n_train), train_prices.values)
te_slope, *_ = _stats.linregress(np.arange(n_test),  test_prices.values)
chow_F, chow_p = _chow_test(train_prices, test_prices)

print("\n─── Structural Break Analysis ─────────────────────────")
print(f"  Train  mean={train_prices.mean():>10,.0f}  "
      f"std={train_prices.std():>8,.0f}  slope={tr_slope:>+8.2f} INR/day")
print(f"  Test   mean={test_prices.mean():>10,.0f}  "
      f"std={test_prices.std():>8,.0f}  slope={te_slope:>+8.2f} INR/day")
print(f"  Chow test: F={chow_F:.4f}, p={chow_p:.4f} "
      f"({'structural break detected' if chow_p < 0.05 else 'no significant break'})")
print("─────────────────────────────────────────────────────")

# ── 3. Estimate K ─────────────────────────────────────────────────────────────
signal_array = price.values.astype(float)
N = len(signal_array)

K_emd = None
try:
    from PyEMD import EMD
    emd = EMD()
    emd.emd(signal_array)
    imfs_emd = emd.get_imfs_and_residue()[0]
    K_emd = imfs_emd.shape[0]
    print(f"\nEMD produced {K_emd} IMFs → using K={K_emd} for VMD")
except ImportError:
    print("\nPyEMD not installed; using log2(N) heuristic for K")
except Exception as e:
    print(f"\nEMD failed ({e}); falling back to heuristic")

if K_emd is not None:
    K = int(np.clip(K_emd, 4, 12))
    print(f"EMD count={K_emd} → VMD K={K}")
else:
    K = int(np.clip(np.floor(np.log2(N)), 6, 12))

print(f"Selected K = {K}  (N={N})")

# ── 4. Run VMD ────────────────────────────────────────────────────────────────
print(f"\nRunning VMD (K={K}, α={ALPHA})… this may take several minutes …")
u, u_hat, omega = VMD(signal_array, ALPHA, TAU, K, DC, INIT, TOL)

print(f"[VMD] raw u.shape={u.shape}")

if u.ndim == 3:
    u = u[-1]
if omega.ndim == 2:
    omega = omega[-1]

sort_idx = np.argsort(omega)
u_sorted     = u[sort_idx, :]
omega_sorted = omega[sort_idx]
print(f"[VMD] omega (sorted): {np.round(omega_sorted, 5)}")

reconstruction = u_sorted.sum(axis=0)
recon_err = np.max(np.abs(signal_array[:len(reconstruction)] - reconstruction))
print(f"Max reconstruction error: {recon_err:.4f} INR/kg "
      f"({recon_err/signal_array.mean()*100:.4f}% of mean)")

# ── Residual correction ───────────────────────────────────────────────────────
residual     = signal_array - u_sorted.sum(axis=0)
residual_pct = np.max(np.abs(residual)) / signal_array.mean() * 100
print(f"Residual max={np.max(np.abs(residual)):.2f} ({residual_pct:.3f}% of mean)")

if residual_pct > VMD_RESIDUAL_THRESHOLD_PCT:
    print(f"  Residual > {VMD_RESIDUAL_THRESHOLD_PCT}% → appending as IMF{K+1}")
    u_sorted = np.vstack([u_sorted, residual.reshape(1, -1)])
    K += 1
    print(f"  Final K after residual append: {K}")
else:
    print(f"  Residual < threshold → reconstruction clean")

final_err = np.max(np.abs(signal_array - u_sorted.sum(axis=0)))
print(f"Final reconstruction error: {final_err:.4f} INR/kg")

np.save(OUT_IMFS, u_sorted)
print(f"Saved: {OUT_IMFS}  (shape={u_sorted.shape})")

# ── 5. Table 5 — IMF Statistics ───────────────────────────────────────────────
print("\nComputing Table 5 statistics…")

rows = []
for i in range(K):
    imf      = u_sorted[i, :]
    omega_val = round(float(omega_sorted[i]), 5) if i < len(omega_sorted) else np.nan
    sig      = signal_array[:len(imf)]

    fft_mag  = np.abs(np.fft.rfft(imf))
    fft_freq = np.fft.rfftfreq(len(imf))
    dom_idx  = np.argmax(fft_mag[1:]) + 1
    dom_freq = fft_freq[dom_idx]
    period_days = round(1.0 / dom_freq, 1) if dom_freq > 0 else np.nan

    rows.append({
        "Mode":             f"IMF{i+1}",
        "Omega":            omega_val,
        "Frequency":        round(dom_freq, 5),
        "Period (days)":    period_days,
        "Variance Ratio":   round(np.var(imf) / np.var(sig), 4),
        "Correlation":      round(np.corrcoef(imf, sig)[0, 1], 4),
        "Mean|IMF|":        round(np.mean(np.abs(imf)), 4),
    })

table5 = pd.DataFrame(rows)
table5.to_csv(OUT_TAB5, index=False)
print("\nTable 5 — IMF Statistics:")
print(table5.to_string(index=False))
print(f"\nSaved: {OUT_TAB5}")

# ── NEW: Variance ratio diagnostic ───────────────────────────────────────────
imf1_vr = float(table5.loc[0, "Variance Ratio"])
print("\n" + "─" * 60)
print(f"DA DIAGNOSTIC — IMF1 variance ratio: {imf1_vr:.4f}")
if imf1_vr < IMF1_VR_THRESHOLD:
    print(
        f"  WARNING: IMF1 variance ratio {imf1_vr:.4f} < threshold {IMF1_VR_THRESHOLD}.\n"
        f"  The trend component is too fragmented. Directional accuracy in\n"
        f"  the downstream ensemble will be severely degraded.\n"
        f"  ACTION REQUIRED: re-run step1 with ALPHA = 4000 (set at top of\n"
        f"  this file) or reduce K by 1 in the EMD-based K selection block."
    )
else:
    print(
        f"  OK: trend IMF captures {imf1_vr*100:.1f}% of signal variance.\n"
        f"  Decomposition quality is sufficient for directional forecasting."
    )
n_low_imfs  = int((table5["Variance Ratio"] > 0.05).sum())
print(f"  IMFs with >5% variance share: {n_low_imfs} of {K}")
print("─" * 60)

# ── 6. Fig 4 — Price with Train/Test Split ────────────────────────────────────
print("\nPlotting Fig 4…")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(train_prices.index, train_prices.values,
        color="#27ae60", linewidth=0.9, label="Training set")
ax.plot(test_prices.index, test_prices.values,
        color="#e74c3c", linewidth=0.9, label="Testing set")
ax.axvline(x=train_prices.index[-1], color="black",
           linestyle="--", linewidth=0.8, alpha=0.6,
           label=f"Split: {train_prices.index[-1].date()}")
ax.set_title("MCX Silver Price — Daily (INR/kg)", fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR/kg)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_FIG4, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG4}")

# ── 7. Fig 7 — Decomposed IMFs ───────────────────────────────────────────────
print("Plotting Fig 7…")

imf_index = price.index[:u_sorted.shape[1]]
fig, axes = plt.subplots(K, 1, figsize=(14, K * 1.6), sharex=True)
if K == 1:
    axes = [axes]

for i in range(K):
    imf = u_sorted[i, :]
    ax  = axes[i]
    ax.plot(imf_index[:n_train], imf[:n_train],
            color="#27ae60", linewidth=0.7)
    ax.plot(imf_index[n_train:len(imf)], imf[n_train:],
            color="#e74c3c", linewidth=0.7)
    ax.set_ylabel(f"IMF{i+1}", fontsize=9, rotation=0,
                  labelpad=40, va="center", ha="right")
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.tick_params(axis="y", labelsize=7)
    vr = table5.loc[i, "Variance Ratio"]
    ax.text(0.99, 0.85, f"VR={vr:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="#555")

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45, fontsize=8)
fig.suptitle(f"VMD Decomposed IMFs (K={K}) — MCX Silver Daily (INR/kg)",
             fontsize=13, fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig(OUT_FIG7, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG7}")

# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 COMPLETE. Files produced:")
print(f"  {OUT_FIG4}       → Fig 4")
print(f"  {OUT_FIG7}  → Fig 7")
print(f"  {OUT_TAB5}     → Table 5")
print(f"  {OUT_IMFS}             → IMFs for Steps 2–6")
print(f"  {OUT_SILVER}  → Silver series for Steps 2–6")
print(f"  {OUT_NTRAIN}  → train split for Steps 3–6")
print("=" * 60)
print("NEXT: python step2_entropy_daily.py")