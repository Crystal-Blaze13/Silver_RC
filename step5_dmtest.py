"""
STEP 5 — Diebold-Mariano Statistical Test
Produces: Table 9
Input:    predictions.pkl
Outputs:  table9_dm_test.csv
"""

import numpy as np
import pandas as pd
import pickle
from scipy import stats

# ── 1. Load predictions ───────────────────────────────────────
print("=" * 55)
print("STEP 5: Diebold-Mariano Test")
print("=" * 55)

with open("predictions.pkl", "rb") as f:
    data = pickle.load(f)

single_preds  = data["single_preds"]
decomp_preds  = data["decomp_preds"]
y_true        = data["y_true_test"]

all_preds = {**single_preds, **decomp_preds}
model_names = list(all_preds.keys())
print(f"Testing {len(model_names)} models: {model_names}")

# ── 2. Diebold-Mariano Test ───────────────────────────────────
def dm_test(y_true, pred1, pred2, h=1):
    """
    Diebold-Mariano test.
    H0: equal predictive accuracy
    H1: model1 is significantly better than model2

    Positive DM statistic → pred1 is BETTER than pred2
    (lower loss = better)

    Returns: (DM statistic, p-value)
    """
    y_true = np.array(y_true)
    e1 = y_true - np.array(pred1)
    e2 = y_true - np.array(pred2)

    # Loss differential using squared error
    d = e1**2 - e2**2   # negative if model1 is better

    # Harvey, Leybourne, Newbold (1997) correction
    n    = len(d)
    d_mean = np.mean(d)

    # Newey-West variance estimate
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k    = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += (1 - k / (h + 1)) * gamma_k
    var_d = (gamma0 + 2 * gamma_sum) / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = -d_mean / np.sqrt(var_d)   # negative because lower loss = better

    # HLN correction factor
    correction = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat    = dm_stat * correction

    # Two-tailed p-value using t-distribution
    p_val = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))

    return round(dm_stat, 4), round(p_val, 4)

def significance_stars(p_val):
    """Return *, **, *** based on p-value."""
    if pd.isna(p_val):
        return ""
    if p_val < 0.01:
        return "***"
    elif p_val < 0.05:
        return "**"
    elif p_val < 0.10:
        return "*"
    return ""

# ── 3. Build DM test matrix ───────────────────────────────────
print("\nRunning pairwise DM tests...")

n_models = len(model_names)
dm_stats = pd.DataFrame(np.nan, index=model_names, columns=model_names)
dm_pvals = pd.DataFrame(np.nan, index=model_names, columns=model_names)
dm_display = pd.DataFrame("", index=model_names, columns=model_names)

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i == j:
            continue
        stat, pval = dm_test(y_true, all_preds[m1], all_preds[m2])
        dm_stats.loc[m1, m2] = stat
        dm_pvals.loc[m1, m2] = pval
        stars = significance_stars(pval)
        if not np.isnan(stat):
            dm_display.loc[m1, m2] = f"{stat:.4f}{stars}"

# ── 4. Save Table 9 ───────────────────────────────────────────
dm_display.to_csv("table9_dm_test.csv")

print("\nTable 9 — DM Test Results (row model vs column model):")
print("Positive = row model better than column model")
print("* p<0.10, ** p<0.05, *** p<0.01\n")

# Print a clean version focusing on proposed method vs others
print("Proposed method vs all others:")
proposed_row = dm_display.loc['Proposed']
for col, val in proposed_row.items():
    if col != 'Proposed' and val != "":
        pval = dm_pvals.loc['Proposed', col]
        print(f"  vs {col:<20}: {val}  (p={pval:.4f})")

print("\nFull matrix saved: table9_dm_test.csv")

print("\n" + "=" * 55)
print("STEP 5 COMPLETE")
print("  table9_dm_test.csv  ← Table 9")
print("=" * 55)
print("NEXT: python3 step6_trading.py")