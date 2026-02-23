"""
STEP 1 — VMD Decomposition of Silver Price
============================================
Replicates:
  - Fig 4  : Silver price time series with train/test split
  - Fig 7  : Decomposed IMFs plot
  - Table 5: IMF frequency, period, variance ratio, correlation

HOW TO RUN:
-----------
1. pip install vmdpy pandas numpy matplotlib scipy
2. Place this script in same folder as master_weekly_prices.csv
3. python3 step1_vmd_decompose.py

OUTPUTS:
  fig4_silver_price_split.png
  fig7_imf_decomposition.png
  table5_imf_statistics.csv
  imfs.npy              <- saved IMFs, used by all later steps
  silver_weekly.csv     <- weekly silver price, used by later steps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from vmdpy import VMD

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
DATA_FILE   = "master_weekly_prices.csv"
TRAIN_RATIO = 0.8      # 80% train, 20% test (same as paper)

# VMD hyperparameters (following paper)
ALPHA = 2000   # bandwidth constraint
TAU   = 0      # noise tolerance
DC    = 0      # no DC component forced
INIT  = 1      # initialize omegas uniformly
TOL   = 1e-7   # convergence tolerance

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: VMD Decomposition")
print("=" * 55)

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
price_weekly = df['silver'].dropna()

print(f"Loaded {len(price_weekly)} weekly observations")
print(f"Date range: {price_weekly.index[0].date()} → {price_weekly.index[-1].date()}")

# Save silver weekly for later steps
price_weekly.to_csv("silver_weekly.csv", header=True)

# ─────────────────────────────────────────────
# 2. Train / Test split (80/20)
# ─────────────────────────────────────────────
n_total = len(price_weekly)
n_train = int(n_total * TRAIN_RATIO)
n_test  = n_total - n_train

train_prices = price_weekly.iloc[:n_train]
test_prices  = price_weekly.iloc[n_train:]

print(f"\nTrain: {n_train} weeks ({train_prices.index[0].date()} → {train_prices.index[-1].date()})")
print(f"Test : {n_test}  weeks ({test_prices.index[0].date()} → {test_prices.index[-1].date()})")

# ─────────────────────────────────────────────
# 3. Estimate K (number of IMFs)
# Paper: uses EMD first to estimate K adaptively
# We use log2(N) heuristic, clamped to 6-10
# ─────────────────────────────────────────────
signal_array = price_weekly.values.astype(float)
N = len(signal_array)
K = int(np.clip(np.floor(np.log2(N)), 6, 10))
print(f"\nEstimated K = {K} IMFs (N={N}, log2(N)={np.log2(N):.2f})")

# ─────────────────────────────────────────────
# 4. Run VMD
# ─────────────────────────────────────────────
print(f"\nRunning VMD (K={K})... this takes ~30-60 seconds...")
u, u_hat, omega = VMD(signal_array, ALPHA, TAU, K, DC, INIT, TOL)

# Sort IMFs from lowest to highest frequency (like paper: IMF1=trend, IMFn=noise)
sort_idx = np.argsort(omega[:, -1])
u_sorted = u[sort_idx, :]

print(f"VMD complete. IMF matrix shape: {u_sorted.shape}")

# Save IMFs for all later steps
np.save("imfs.npy", u_sorted)
print("Saved: imfs.npy")

# ─────────────────────────────────────────────
# 5. TABLE 5 — IMF Statistics
# ─────────────────────────────────────────────
print("\nComputing Table 5 statistics...")

rows = []
for i in range(K):
    imf = u_sorted[i, :]

    # Dominant frequency via FFT
    fft_mag  = np.abs(np.fft.rfft(imf))
    fft_freq = np.fft.rfftfreq(len(imf))
    dom_freq = fft_freq[np.argmax(fft_mag[1:]) + 1]

    # Period in weeks
    period = round(1.0 / dom_freq, 4) if dom_freq > 0 else np.nan

    # Variance ratio
    var_ratio = round(np.var(imf) / np.var(signal_array), 4)

    # Correlation with original
    corr = round(np.corrcoef(imf, signal_array)[0, 1], 4)

    rows.append({
        "Mode":           f"IMF{i+1}",
        "Frequency":      round(dom_freq, 4),
        "Period (weeks)": period,
        "Variance Ratio": var_ratio,
        "Correlation":    corr,
    })

table5 = pd.DataFrame(rows)
table5.to_csv("table5_imf_statistics.csv", index=False)
print("\nTable 5:")
print(table5.to_string(index=False))
print("\nSaved: table5_imf_statistics.csv")

# ─────────────────────────────────────────────
# 6. FIG 4 — Silver Price with Train/Test Split
# ─────────────────────────────────────────────
print("\nPlotting Fig 4...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train_prices.index, train_prices.values,
        color='#27ae60', linewidth=1.2, label='Training set')
ax.plot(test_prices.index, test_prices.values,
        color='#e74c3c', linewidth=1.2, label='Testing set')
ax.axvline(x=train_prices.index[-1], color='black',
           linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('Silver Price Time Series (Weekly)', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD/oz)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig4_silver_price_split.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_silver_price_split.png")

# ─────────────────────────────────────────────
# 7. FIG 7 — Decomposed IMFs
# ─────────────────────────────────────────────
print("Plotting Fig 7...")

fig, axes = plt.subplots(K, 1, figsize=(14, K * 1.8), sharex=True)
for i in range(K):
    imf = u_sorted[i, :]
    ax  = axes[i]
    # Train portion in green, test in red (same as paper)
    ax.plot(price_weekly.index[:n_train], imf[:n_train],
            color='#27ae60', linewidth=0.9)
    ax.plot(price_weekly.index[n_train:], imf[n_train:],
            color='#e74c3c', linewidth=0.9)
    ax.set_ylabel(f'IMF{i+1}', fontsize=9, rotation=0,
                  labelpad=35, va='center', ha='right')
    ax.axhline(0, color='grey', linewidth=0.4, linestyle='--')
    ax.tick_params(axis='y', labelsize=7)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
plt.xticks(rotation=45, fontsize=8)
fig.suptitle('VMD Decomposed IMFs — Silver Price', fontsize=13,
             fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig("fig7_imf_decomposition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig7_imf_decomposition.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 1 COMPLETE. Files produced:")
print("  fig4_silver_price_split.png  → Fig 4")
print("  fig7_imf_decomposition.png   → Fig 7")
print("  table5_imf_statistics.csv    → Table 5")
print("  imfs.npy                     → input for Steps 2-6")
print("  silver_weekly.csv            → input for Steps 2-6")
print("=" * 55)
print("NEXT: Run step2_entropy.py")