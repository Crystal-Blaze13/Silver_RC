"""
STEP 2 — Approximate Entropy & Complexity Classification
Produces: Fig 8, saves low/high complexity IMF labels
Input:    imfs.npy, silver_weekly.csv
Outputs:  fig8_approximate_entropy.png
          imf_complexity.csv  (used by steps 3-6)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────
IMF_FILE    = "imfs.npy"
SILVER_FILE = "silver_weekly.csv"
AE_M        = 2      # embedding dimension (standard value)
AE_R_COEF   = 0.2   # r = 0.2 * std(series) — standard in literature

# ── 1. Load data ───────────────────────────────────────────────
print("=" * 55)
print("STEP 2: Approximate Entropy")
print("=" * 55)

u_sorted     = np.load(IMF_FILE)
K            = u_sorted.shape[0]
silver_df    = pd.read_csv(SILVER_FILE, index_col=0, parse_dates=True)
signal_array = silver_df.iloc[:, 0].values.astype(float)

print(f"Loaded {K} IMFs")

# ── 2. Approximate Entropy function ───────────────────────────
def approximate_entropy(U, m, r):
    """
    Compute Approximate Entropy (ApEn) of a time series U.

    Parameters:
        U : 1D numpy array — the time series
        m : int — embedding dimension (typically 2)
        r : float — tolerance (typically 0.2 * std(U))

    Returns:
        ApEn value (float) — higher = more complex/irregular
    """
    N = len(U)

    def phi(m):
        # Build template vectors of length m
        templates = np.array([U[i:i+m] for i in range(N - m + 1)])
        # Count matches for each template
        counts = []
        for i in range(len(templates)):
            # Distance = max absolute difference (Chebyshev distance)
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            # Count how many are within tolerance r
            counts.append(np.sum(dist <= r) / (N - m + 1))
        # Return average log of counts
        return np.mean(np.log(counts))

    return phi(m) - phi(m + 1)

# ── 3. Compute ApEn for original series and all IMFs ──────────
print("\nComputing Approximate Entropy (this may take 1-2 minutes)...")

# ApEn of original signal (used as threshold)
r_orig = AE_R_COEF * np.std(signal_array)
ae_orig = approximate_entropy(signal_array, AE_M, r_orig)
print(f"  Original series ApEn: {ae_orig:.4f}")

# ApEn of each IMF
ae_values = []
for i in range(K):
    imf = u_sorted[i, :]
    r   = AE_R_COEF * np.std(imf)
    ae  = approximate_entropy(imf, AE_M, r)
    ae_values.append(ae)
    print(f"  IMF{i+1} ApEn: {ae:.4f}")

ae_values = np.array(ae_values)

# ── 4. Classify IMFs as low or high complexity ────────────────
# Paper: IMFs with ApEn BELOW original = low complexity
#        IMFs with ApEn ABOVE original = high complexity
complexity = []
for i in range(K):
    if ae_values[i] < ae_orig:
        complexity.append("Low")
    else:
        complexity.append("High")

# Build summary dataframe
imf_complexity = pd.DataFrame({
    "IMF":        [f"IMF{i+1}" for i in range(K)],
    "ApEn":       np.round(ae_values, 4),
    "Complexity": complexity,
})
imf_complexity.to_csv("imf_complexity.csv", index=False)

print("\nIMF Complexity Classification:")
print(imf_complexity.to_string(index=False))

low_imfs  = [i+1 for i, c in enumerate(complexity) if c == "Low"]
high_imfs = [i+1 for i, c in enumerate(complexity) if c == "High"]
print(f"\nLow complexity  IMFs (→ ARIMA) : {low_imfs}")
print(f"High complexity IMFs (→ LSTM)  : {high_imfs}")
print("Saved: imf_complexity.csv")

# ── 5. FIG 8 — Approximate Entropy Bar Chart ──────────────────
print("\nPlotting Fig 8...")

imf_labels = [f"IMF{i+1}" for i in range(K)]
colors     = ['#3498db' if c == "Low" else '#e74c3c' for c in complexity]

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(imf_labels, ae_values, color=colors, edgecolor='white',
              linewidth=0.5, zorder=3)

# Dashed line = original series entropy (threshold)
ax.axhline(y=ae_orig, color='black', linestyle='--', linewidth=1.5,
           label=f'Original series ApEn = {ae_orig:.4f}', zorder=4)

# Annotate bars with values
for bar, val in zip(bars, ae_values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Legend patches for colours
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Low complexity → ARIMA'),
    Patch(facecolor='#e74c3c', label='High complexity → LSTM'),
    plt.Line2D([0], [0], color='black', linestyle='--',
               label=f'Original ApEn = {ae_orig:.4f}')
]
ax.legend(handles=legend_elements, fontsize=9)

ax.set_title('Approximate Entropy Values of Decomposed IMFs',
             fontsize=13, fontweight='bold')
ax.set_xlabel('IMF Component', fontsize=11)
ax.set_ylabel('Approximate Entropy', fontsize=11)
ax.set_ylim(0, max(ae_values) * 1.2)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig("fig8_approximate_entropy.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig8_approximate_entropy.png")

print("\n" + "=" * 55)
print("STEP 2 COMPLETE")
print("  fig8_approximate_entropy.png")
print("  imf_complexity.csv  ← needed by steps 3-6")
print(f"  Low complexity  → ARIMA : IMFs {low_imfs}")
print(f"  High complexity → LSTM  : IMFs {high_imfs}")
print("=" * 55)
print("NEXT: python3 step3_lasso.py")