"""
STEP 2 (daily) — Approximate Entropy & Complexity Classification
================================================================
FIX applied (DA diagnostic):
  - After entropy classification, checks that at least MIN_LOW_IMFS IMFs
    are labelled "Low" (ARIMA).  For a commodity like silver the trend
    and business-cycle components must go to ARIMA; if the entropy
    threshold assigns everything to LSTM, direction degrades sharply.
  - Adds a MIN_LOW_IMFS guard: if fewer than 2 IMFs clear the threshold
    the lowest-entropy IMFs are promoted to "Low" regardless.
  - Prints a clear per-IMF variance ratio so you can cross-check with
    table5_imf_statistics.csv to verify which IMFs dominate variance.

Input:  imfs_daily.npy, financial_data/processed/silver_daily.csv
Output: fig8_approximate_entropy.png
        imf_complexity_daily.csv  (used by steps 3–6)

HOW TO RUN (from the daily/ directory):
  python step2_entropy_daily.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────────────────────
IMF_FILE    = "../processed/imfs_daily.npy"
SILVER_FILE = "../processed/silver_daily.csv"
AE_M        = 2
AE_R_COEF   = 0.2
SAMPEN_M    = 2
SAMPEN_R    = 0.2

# Minimum number of IMFs that must be classified as Low complexity (ARIMA).
# For a trending commodity, trend + business-cycle IMFs should always be
# modelled by ARIMA.  Set to 2 to match the carbon paper's split pattern.
MIN_LOW_IMFS = 2

OUT_FIG8  = "../results/figures/fig8_approximate_entropy.png"
OUT_COMP  = "../processed/imf_complexity_daily.csv"

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 (daily): Approximate Entropy & Complexity Classification")
print("=" * 60)

u_sorted     = np.load(IMF_FILE)
K            = u_sorted.shape[0]
silver_df    = pd.read_csv(SILVER_FILE, index_col=0, parse_dates=True)
signal_array = silver_df.iloc[:, 0].values.astype(float)

print(f"Loaded {K} IMFs, N={u_sorted.shape[1]} samples")

# Print variance ratios for cross-reference with step1 table5
print("\nVariance ratios (for cross-check with table5_imf_statistics.csv):")
total_var = np.var(signal_array)
for i in range(K):
    vr = np.var(u_sorted[i, :]) / total_var
    print(f"  IMF{i+1:02d}  VR={vr:.4f}  ({'dominant trend' if vr > 0.5 else 'secondary' if vr > 0.05 else 'noise'})")


# ── 2. Approximate Entropy (Pincus 1991) ─────────────────────────────────────
def approximate_entropy(U: np.ndarray, m: int, r: float) -> float:
    U = np.asarray(U, dtype=float)
    if np.std(U) == 0:
        return np.nan
    N = len(U)

    def _phi(m_):
        templates = np.lib.stride_tricks.sliding_window_view(U, m_)
        C = np.zeros(len(templates))
        for j in range(len(templates)):
            dist = np.max(np.abs(templates - templates[j]), axis=1)
            C[j] = np.sum(dist <= r) / len(templates)
        return np.mean(np.log(np.maximum(C, 1e-300)))

    return float(_phi(m) - _phi(m + 1))


# ── 3. Sample Entropy (Richman & Moorman 2000) ───────────────────────────────
def sample_entropy(U: np.ndarray, m: int, r: float) -> float:
    U = np.asarray(U, dtype=float)
    if np.std(U) == 0:
        return np.nan
    N = len(U)

    def _count(m_):
        templates = np.lib.stride_tricks.sliding_window_view(U, m_)
        count = 0
        for j in range(len(templates)):
            dist = np.max(np.abs(templates - templates[j]), axis=1)
            dist[j] = np.inf
            count += np.sum(dist <= r)
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return np.nan
    return float(-np.log(A / B))


# ── 4. Compute entropies ──────────────────────────────────────────────────────
print("\nComputing entropies (may take 5–15 min for daily N)…")

r_orig  = AE_R_COEF * np.std(signal_array)
ae_orig = approximate_entropy(signal_array, AE_M, r_orig)
se_orig = sample_entropy(signal_array, SAMPEN_M, SAMPEN_R * np.std(signal_array))
print(f"  Original series  ApEn={ae_orig:.4f}  SampEn={se_orig:.4f}")

ae_values = []
se_values = []
for i in range(K):
    imf = u_sorted[i, :]
    r_i = AE_R_COEF * np.std(imf)
    ae  = approximate_entropy(imf, AE_M, r_i)
    se  = sample_entropy(imf, SAMPEN_M, SAMPEN_R * np.std(imf))
    ae_values.append(ae)
    se_values.append(se)
    print(f"  IMF{i+1:02d}   ApEn={ae:.4f}   SampEn={se:.4f}")

ae_values = np.array(ae_values)
se_values = np.array(se_values)


# ── 5. Classify IMFs ──────────────────────────────────────────────────────────
complexity = ["Low" if v < ae_orig else "High" for v in ae_values]

# Guard: ensure at least MIN_LOW_IMFS are labelled Low.
# If not enough IMFs clear the threshold naturally, promote the IMFs with
# the lowest entropy values (most predictable) to "Low".
n_low = sum(1 for c in complexity if c == "Low")
if n_low < MIN_LOW_IMFS:
    # Sort IMFs by entropy ascending; promote the top ones until we hit the minimum
    finite_mask = np.isfinite(ae_values)
    sorted_idx  = np.argsort(np.where(finite_mask, ae_values, np.inf))
    promoted    = 0
    for idx in sorted_idx:
        if complexity[idx] == "High":
            complexity[idx] = "Low"
            promoted += 1
            print(
                f"  PROMOTED IMF{idx+1} to Low (ApEn={ae_values[idx]:.4f}) "
                f"— MIN_LOW_IMFS={MIN_LOW_IMFS} guard applied"
            )
        if sum(1 for c in complexity if c == "Low") >= MIN_LOW_IMFS:
            break
    if promoted:
        print(
            f"  NOTE: {promoted} extra IMF(s) promoted to Low complexity.\n"
            f"  The original entropy threshold would have sent them all to LSTM,\n"
            f"  which degrades directional accuracy on trending commodities."
        )

imf_complexity = pd.DataFrame({
    "IMF":        [f"IMF{i+1}" for i in range(K)],
    "ApEn":       np.round(ae_values, 4),
    "SampEn":     np.round(se_values, 4),
    "Complexity": complexity,
})
imf_complexity.to_csv(OUT_COMP, index=False)

print("\nIMF Complexity Classification:")
print(imf_complexity.to_string(index=False))
print(f"\nOriginal series ApEn = {ae_orig:.4f}  (threshold)")
low_imfs  = [i+1 for i, c in enumerate(complexity) if c == "Low"]
high_imfs = [i+1 for i, c in enumerate(complexity) if c == "High"]
print(f"Low  complexity → ARIMA : IMFs {low_imfs}")
print(f"High complexity → LSTM  : IMFs {high_imfs}")
print(f"Saved: {OUT_COMP}")


# ── 6. Fig 8 ─────────────────────────────────────────────────────────────────
print("\nPlotting Fig 8…")

imf_labels = [f"IMF{i+1}" for i in range(K)]
colors     = ["#3498db" if c == "Low" else "#e74c3c" for c in complexity]

fig, ax1 = plt.subplots(figsize=(max(11, K * 0.9), 5))

bars = ax1.bar(imf_labels, ae_values, color=colors,
               edgecolor="white", linewidth=0.5, zorder=3, alpha=0.85)
ax1.axhline(y=ae_orig, color="black", linestyle="--", linewidth=1.6,
            label=f"Original ApEn = {ae_orig:.4f}", zorder=4)

for bar, val in zip(bars, ae_values):
    if np.isfinite(val):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(ae_values[np.isfinite(ae_values)]) * 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax2 = ax1.twinx()
valid_se = np.where(np.isfinite(se_values), se_values, 0)
ax2.plot(imf_labels, valid_se, color="#f39c12", marker="D",
         markersize=6, linewidth=1.4, linestyle="-.",
         label=f"SampEn (orig={se_orig:.4f})", zorder=5)
ax2.axhline(y=se_orig, color="#f39c12", linestyle=":", linewidth=1.2, alpha=0.7)
ax2.set_ylabel("Sample Entropy", fontsize=10, color="#f39c12")
ax2.tick_params(axis="y", labelcolor="#f39c12")

legend_elements = [
    Patch(facecolor="#3498db", alpha=0.85, label="Low complexity → ARIMA"),
    Patch(facecolor="#e74c3c", alpha=0.85, label="High complexity → LSTM"),
    Line2D([0], [0], color="black", linestyle="--",
           label=f"Original ApEn = {ae_orig:.4f}"),
    Line2D([0], [0], color="#f39c12", linestyle="-.", marker="D",
           markersize=5, label=f"SampEn (orig={se_orig:.4f})"),
]
ax1.legend(handles=legend_elements, fontsize=9, loc="upper right")
ax1.set_xlabel("IMF Mode", fontsize=10)
ax1.set_ylabel("Approximate Entropy", fontsize=10)
ax1.set_title("Fig 8 — IMF Approximate Entropy (Daily MCX Silver)",
              fontsize=12, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(OUT_FIG8, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG8}")

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print(f"  {OUT_COMP}   ← complexity labels")
print(f"  {OUT_FIG8}")
print("=" * 60)
print("NEXT: python step3_lasso_daily.py")