"""
STEP 5 — Diebold-Mariano Statistical Test
==========================================
Produces: Table 9
Input:    predictions.pkl
Outputs:  table9_dm_test.csv
          fig_dm_heatmap.png   (visual matrix)

IMPROVEMENTS OVER ORIGINAL:
  - Harvey-Leybourne-Newbold (1997) finite-sample correction is applied
    correctly: the correction factor uses the actual n and h, not a
    mis-specified constant.
  - Newey-West variance uses a data-driven lag (Bartlett kernel with
    bandwidth h = int(n^(1/3)) rather than fixed h=1).
  - Both one-sided (row model better than column) and two-sided p-values
    are reported.
  - DM matrix exported as a colour-coded heat-map (Fig) for visual inspection.
  - Skips pairs with < 10 test observations (degenerate case).
  - Handles NaN predictions gracefully.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings("ignore")
from scipy import stats

OUT_TABLE9 = "../results/tables/table9_dm_test.csv"
OUT_FIG    = "../results/figures/fig_dm_heatmap.png"

# ── 1. Load predictions ───────────────────────────────────────
print("=" * 60)
print("STEP 5: Diebold-Mariano Test")
print("=" * 60)

with open("../processed/predictions.pkl", "rb") as f:
    data = pickle.load(f)

single_preds  = data["single_preds"]
decomp_preds  = data["decomp_preds"]
y_true        = data["y_true_test"]

all_preds   = {**single_preds, **decomp_preds}
model_names = list(all_preds.keys())
n           = len(y_true)
print(f"Test set: {n} observations")
print(f"Models  : {model_names}")

# ── 2. Diebold-Mariano test (HLN 1997) ───────────────────────
def dm_test(y_true, pred1, pred2, h=1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) correction.

    Returns:
        dm_stat  — positive means pred1 is BETTER (lower squared loss)
        p_1sided — one-sided p-value (H1: pred1 better than pred2)
        p_2sided — two-sided p-value (H1: unequal predictive accuracy)
    """
    y = np.array(y_true, dtype=float)
    e1 = y - np.array(pred1, dtype=float)
    e2 = y - np.array(pred2, dtype=float)

    if np.any(np.isnan(e1)) or np.any(np.isnan(e2)):
        return np.nan, np.nan, np.nan

    # Loss differential: negative if model1 is better
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 10:
        return np.nan, np.nan, np.nan

    d_bar = np.mean(d)

    # Newey-West variance with data-driven bandwidth (Bartlett kernel)
    bw = max(1, int(n ** (1.0 / 3.0)))
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, bw + 1):
        gamma_k    = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += (1.0 - k / (bw + 1)) * gamma_k
    var_d = (gamma0 + 2.0 * gamma_sum) / n

    if var_d <= 0:
        return np.nan, np.nan, np.nan

    # Raw DM statistic (negative = model1 better, since lower loss is better)
    dm_raw = d_bar / np.sqrt(var_d)

    # HLN finite-sample correction factor
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat    = -dm_raw * correction        # flip sign: positive = model1 better

    df     = n - 1
    p_1sid = 1.0 - stats.t.cdf(dm_stat, df=df)        # one-sided: H1 = model1 better
    p_2sid = 2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=df))

    return round(float(dm_stat), 4), round(p_1sid, 4), round(p_2sid, 4)

def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""

# ── 3. Build pairwise DM matrices ────────────────────────────
print("\nRunning pairwise DM tests…")

nm = len(model_names)
dm_stats  = pd.DataFrame(np.nan, index=model_names, columns=model_names)
dm_p1sid  = pd.DataFrame(np.nan, index=model_names, columns=model_names)
dm_p2sid  = pd.DataFrame(np.nan, index=model_names, columns=model_names)
dm_display = pd.DataFrame("",   index=model_names, columns=model_names)

for m1 in model_names:
    for m2 in model_names:
        if m1 == m2:
            continue
        stat, p1, p2 = dm_test(y_true, all_preds[m1], all_preds[m2])
        dm_stats.loc[m1, m2]  = stat
        dm_p1sid.loc[m1, m2]  = p1
        dm_p2sid.loc[m1, m2]  = p2
        stars = sig_stars(p1)
        if not np.isnan(stat):
            dm_display.loc[m1, m2] = f"{stat:.4f}{stars}"

# ── 4. Print readable comparison ─────────────────────────────
print("\nTable 9 — Proposed vs all other models:")
print("  Positive stat = Proposed is BETTER; stars = one-sided significance")
if 'Proposed' in model_names:
    for col in model_names:
        if col == 'Proposed':
            continue
        val  = dm_display.loc['Proposed', col]
        p1   = dm_p1sid.loc['Proposed', col]
        p2   = dm_p2sid.loc['Proposed', col]
        if val:
            print(f"  vs {col:<22} {val:<14}  p(1-sided)={p1:.4f}  p(2-sided)={p2:.4f}")

print("\nFull DM matrix (row model vs column model):")
print(dm_display.to_string())

# ── 5. Save Table 9 ───────────────────────────────────────────
dm_display.to_csv(OUT_TABLE9)
print(f"\nSaved: {OUT_TABLE9}")

# Also save numeric matrices for downstream use
dm_stats.to_csv("../results/tables/table9_dm_stats_numeric.csv")
dm_p1sid.to_csv("../results/tables/table9_dm_pvals_onesided.csv")
dm_p2sid.to_csv("../results/tables/table9_dm_pvals_twosided.csv")

# ── 6. FIG — DM heat-map ─────────────────────────────────────
print("Plotting DM heat-map…")

stat_vals = dm_stats.values.astype(float)

fig, ax = plt.subplots(figsize=(max(8, nm * 0.9), max(6, nm * 0.9)))

# Diverging colourmap: red = row better, blue = column better
vmax = np.nanpercentile(np.abs(stat_vals), 95)
im   = ax.imshow(stat_vals, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
plt.colorbar(im, ax=ax, label='DM statistic (+ = row better)')

ax.set_xticks(range(nm))
ax.set_yticks(range(nm))
ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(model_names, fontsize=9)

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i == j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=True, color='lightgray'))
            continue
        v  = stat_vals[i, j]
        p  = dm_p1sid.values[i, j]
        s  = sig_stars(p) if not np.isnan(p) else ""
        txt = f"{v:.2f}{s}" if not np.isnan(v) else ""
        ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                color='white' if abs(v) > vmax * 0.6 else 'black')

ax.set_title("DM Test Matrix — MCX Silver Forecasting\n"
             "(+ = row model outperforms column model; *, **, *** = 10/5/1%)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_FIG}")

print("\n" + "=" * 60)
print("STEP 5 COMPLETE")
print(f"  {OUT_TABLE9}   ← Table 9")
print(f"  {OUT_FIG}")
print("=" * 60)
print("NEXT: python3 step6_trading.py")