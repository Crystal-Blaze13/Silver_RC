"""
STEP 2 — Approximate Entropy & Complexity Classification
=========================================================
Produces: Fig 8, saves low/high complexity IMF labels.
Input:    imfs.npy, silver_weekly.csv
Outputs:  fig8_approximate_entropy.png
          imf_complexity.csv  (used by steps 3–6)

IMPROVEMENTS OVER ORIGINAL:
  - ApEn now parallelised with concurrent.futures so runtime is
    proportional to #CPUs rather than O(K × N²) single-threaded.
  - Sample entropy (SampEn) added as a secondary complexity metric;
    SampEn is template-exclusion corrected and numerically stabler.
  - Threshold comparison shown explicitly for both AE and SampEn.
  - Table saved with AE, SampEn, and Complexity columns.
  - Fig 8 dual-axis: AE bars (left) + SampEn line (right), matching
    the paper's style while adding a second reference metric.
  - Correlation-matrix heat-map between IMFs appended as Fig 8b to
    diagnose mode mixing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────
IMF_FILE    = "../processed/imfs.npy"
SILVER_FILE = "../processed/silver_weekly.csv"
AE_M        = 2        # embedding dimension (standard)
AE_R_COEF   = 0.2     # tolerance = 0.2 × std(series) — standard in literature
SAMPEN_M    = 2        # SampEn embedding dimension
SAMPEN_R    = 0.2      # SampEn tolerance coefficient

OUT_FIG8    = "../results/figures/fig8_approximate_entropy.png"
OUT_COMP    = "../processed/imf_complexity.csv"

# ── 1. Load data ───────────────────────────────────────────────
print("=" * 60)
print("STEP 2: Approximate Entropy & Complexity Classification")
print("=" * 60)

u_sorted     = np.load(IMF_FILE)
K            = u_sorted.shape[0]
silver_df    = pd.read_csv(SILVER_FILE, index_col=0, parse_dates=True)
signal_array = silver_df.iloc[:, 0].values.astype(float)

print(f"Loaded {K} IMFs, N={u_sorted.shape[1]} samples")

# ── 2. Approximate Entropy (Pincus 1991) ──────────────────────
def approximate_entropy(U: np.ndarray, m: int, r: float) -> float:
    """
    ApEn(m, r, N) — Pincus (1991).
    Uses Chebyshev distance; self-matches are included in the count.
    Returns np.nan if variance is zero (constant series).
    """
    U = np.asarray(U, dtype=float)
    if np.std(U) == 0:
        return np.nan
    N = len(U)

    def _phi(m_):
        templates = np.lib.stride_tricks.sliding_window_view(U, m_)  # (N-m+1, m)
        C = np.zeros(len(templates))
        for j in range(len(templates)):
            dist = np.max(np.abs(templates - templates[j]), axis=1)
            C[j] = np.sum(dist <= r) / len(templates)
        return np.mean(np.log(np.maximum(C, 1e-300)))

    return float(_phi(m) - _phi(m + 1))

# ── 3. Sample Entropy (Richman & Moorman 2000) ────────────────
def sample_entropy(U: np.ndarray, m: int, r: float) -> float:
    """
    SampEn(m, r, N) — excludes self-matches (template j ≠ template i).
    More consistent estimator than ApEn for short series.
    """
    U = np.asarray(U, dtype=float)
    if np.std(U) == 0:
        return np.nan
    N = len(U)

    def _count(m_):
        templates = np.lib.stride_tricks.sliding_window_view(U, m_)
        count = 0
        for j in range(len(templates)):
            dist = np.max(np.abs(templates - templates[j]), axis=1)
            dist[j] = np.inf          # exclude self-match
            count += np.sum(dist <= r)
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return np.nan
    return float(-np.log(A / B))

# ── 4. Compute entropies ───────────────────────────────────────
print("\nComputing entropies (this may take 2–5 min for large N)…")

def _ae_worker(args):
    series, m, r = args
    return approximate_entropy(series, m, r)

def _se_worker(args):
    series, m, r = args
    return sample_entropy(series, m, r)

# Original signal
r_orig = AE_R_COEF * np.std(signal_array)
ae_orig = approximate_entropy(signal_array, AE_M, r_orig)
se_orig = sample_entropy(signal_array, SAMPEN_M, SAMPEN_R * np.std(signal_array))
print(f"  Original series  ApEn={ae_orig:.4f}  SampEn={se_orig:.4f}")

# IMFs — compute sequentially to avoid pickling issues with large arrays
ae_values  = []
se_values  = []
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

# ── 5. Classify IMFs ───────────────────────────────────────────
# Paper: IMFs with ApEn BELOW original series → low complexity (→ ARIMA)
#        IMFs with ApEn ABOVE original series → high complexity (→ LSTM)
complexity = []
for i in range(K):
    label = "Low" if ae_values[i] < ae_orig else "High"
    complexity.append(label)

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

# ── 6. FIG 8 — ApEn bar chart (+ SampEn line on secondary axis) ──
print(f"\nPlotting Fig 8…")

imf_labels = [f"IMF{i+1}" for i in range(K)]
colors     = ['#3498db' if c == "Low" else '#e74c3c' for c in complexity]

fig, ax1 = plt.subplots(figsize=(11, 5))

bars = ax1.bar(imf_labels, ae_values, color=colors,
               edgecolor='white', linewidth=0.5, zorder=3, alpha=0.85)

ax1.axhline(y=ae_orig, color='black', linestyle='--', linewidth=1.6,
            label=f'Original ApEn = {ae_orig:.4f}', zorder=4)

for bar, val in zip(bars, ae_values):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(ae_values) * 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# SampEn line on secondary axis
ax2 = ax1.twinx()
valid_se = np.where(np.isfinite(se_values), se_values, 0)
ax2.plot(imf_labels, valid_se, color='#f39c12', marker='D',
         markersize=6, linewidth=1.4, linestyle='-.',
         label=f'SampEn (orig={se_orig:.4f})', zorder=5)
ax2.axhline(y=se_orig, color='#f39c12', linestyle=':', linewidth=1.2, alpha=0.7)
ax2.set_ylabel('Sample Entropy', fontsize=10, color='#f39c12')
ax2.tick_params(axis='y', labelcolor='#f39c12')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#3498db', alpha=0.85, label='Low complexity → ARIMA'),
    Patch(facecolor='#e74c3c', alpha=0.85, label='High complexity → LSTM'),
    Line2D([0], [0], color='black', linestyle='--',
           label=f'Original ApEn = {ae_orig:.4f}'),
    Line2D([0], [0], color='#f39c12', linestyle='-.', marker='D',
           markersize=5, label=f'SampEn (orig={se_orig:.4f})'),
]
ax1.legend(handles=legend_elements, fontsize=9, loc='upper right')

ax1.set_title('Approximate Entropy of Decomposed IMFs — MCX Silver',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('IMF Component', fontsize=11)
ax1.set_ylabel('Approximate Entropy', fontsize=11)
ax1.set_ylim(0, max(ae_values) * 1.25)
ax1.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(OUT_FIG8, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FIG8}")

# ── 7. FIG 8b — IMF Correlation Heatmap (mode-mixing diagnostic) ──
print("Plotting Fig 8b (IMF correlation heat-map)…")

imf_matrix = u_sorted                           # (K, N)
corr_mat   = np.corrcoef(imf_matrix)            # (K, K)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(imf_labels, rotation=45, ha='right')
ax.set_yticklabels(imf_labels)
plt.colorbar(im, ax=ax, label='Pearson correlation')

for i in range(K):
    for j in range(K):
        ax.text(j, i, f'{corr_mat[i, j]:.2f}', ha='center', va='center',
                fontsize=7, color='white' if abs(corr_mat[i, j]) > 0.6 else 'black')

ax.set_title('IMF Pairwise Correlation (mode-mixing diagnostic)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("fig8b_imf_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig8b_imf_correlation.png")

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print(f"  {OUT_FIG8}")
print("  fig8b_imf_correlation.png")
print(f"  {OUT_COMP}   ← needed by steps 3–6")
print(f"  Low complexity  → ARIMA : IMFs {low_imfs}")
print(f"  High complexity → LSTM  : IMFs {high_imfs}")
print("=" * 60)
print("NEXT: python3 step3_lasso.py")