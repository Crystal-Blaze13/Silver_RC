"""
STEP 6 — Trading Strategy & Interval Forecasting
Produces: Fig 11, Fig 12, Fig 13, Table 10, Table 11, Table 12
Input:    predictions.pkl, silver_weekly.csv, n_train.npy
Outputs:  table10_interval_errors.csv
          table11_decomp_trading.csv
          table12_single_trading.csv
          fig11_interval_forecasts.png
          fig12_trading_strategy_illustration.png
          fig13_trading_evaluation.png

External features lagged by 1 period to prevent look-ahead bias
— forecasting t using data up to t-1.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────
TRANSACTION_COST = 0.0005   # 0.05% per trade (same as paper)
INITIAL_CAPITAL  = 1.0      # normalised to 1

# ── 1. Load data ───────────────────────────────────────────────
print("=" * 55)
print("STEP 6: Trading Strategy")
print("=" * 55)

with open("predictions.pkl", "rb") as f:
    data = pickle.load(f)

single_preds  = data["single_preds"]
decomp_preds  = data["decomp_preds"]
y_true        = data["y_true_test"]
test_dates    = data["test_dates"]
proposed_pred = data["proposed_pred"]
n_train       = data["n_train"]

silver_df     = pd.read_csv("silver_weekly.csv", index_col=0, parse_dates=True)
silver_all    = silver_df.iloc[:, 0].values

all_preds = {**single_preds, **decomp_preds}
n_test    = len(y_true)

# ── 2. Interval Forecasting ───────────────────────────────────
# Asymmetric ±3% band around the point forecast.
# Upper = forecast × 1.03, Lower = forecast × 0.97.
# Interval constraint (Schemes 1', 2'): if the actual price from
# the prior week falls outside this band, I_t = 0 (no trade that week).

print("\nGenerating interval forecasts (Proposed method)...")

interval_upper = proposed_pred * 1.03
interval_lower = proposed_pred * 0.97

# ── 3. TABLE 10 — Interval Forecast Errors ────────────────────
# Metrics: U (Theil), ARV, RMSDE, CR (Coverage Ratio)

def interval_metrics(y_true, lower, upper):
    """
    Compute interval forecast evaluation metrics.
    U   = Theil statistic (lower = better)
    ARV = Average Relative Variance (lower = better)
    CR  = Coverage Ratio (higher = better, ideally ~0.95)
    """
    y   = np.array(y_true)
    lo  = np.array(lower)
    hi  = np.array(upper)
    n   = len(y)

    # Coverage Ratio: fraction of actual values inside interval
    inside = ((y >= lo) & (y <= hi))
    cr     = inside.mean()

    # Interval width average
    avg_width = np.mean(hi - lo)

    # Winkler score (penalises width + misses)
    alpha  = 0.05
    winkler = []
    for i in range(n):
        w = hi[i] - lo[i]
        if y[i] < lo[i]:
            w += (2 / alpha) * (lo[i] - y[i])
        elif y[i] > hi[i]:
            w += (2 / alpha) * (y[i] - hi[i])
        winkler.append(w)
    winkler_score = np.mean(winkler)

    # Theil U-like statistic
    naive_width = np.std(y)
    U = avg_width / naive_width if naive_width > 0 else np.nan

    # ARV
    mid     = (lo + hi) / 2
    ARV     = np.mean((y - mid)**2) / np.var(y)

    # RMSDE
    RMSDE   = np.sqrt(np.mean((y - mid)**2))

    return {
        "U":      round(U, 4),
        "ARV":    round(ARV, 4),
        "RMSDE":  round(RMSDE, 4),
        "CR":     round(cr, 4),
    }

int_metrics = interval_metrics(y_true, interval_lower, interval_upper)
table10 = pd.DataFrame([int_metrics], index=["Silver (Proposed)"])
table10.to_csv("table10_interval_errors.csv")
print("\nTable 10 — Interval Forecast Errors:")
print(table10.to_string())
print("Saved: table10_interval_errors.csv")

# ── 4. Trading Strategy Logic ─────────────────────────────────
def run_trading_strategy(y_true, y_pred, scheme=1,
                         interval_lower=None, interval_upper=None,
                         tc=TRANSACTION_COST):
    """
    Simulate trading strategy.

    Scheme 1 : Buy if predicted return > 0, sell if < 0
    Scheme 1': Scheme 1 with interval constraint
    Scheme 2 : Only trade if predicted return > transaction cost
    Scheme 2': Scheme 2 with interval constraint

    Returns dict of performance metrics.
    """
    y      = np.array(y_true)
    y_hat  = np.array(y_pred)
    n      = len(y)

    # Predicted returns
    pred_returns = np.diff(y_hat) / y_hat[:-1]
    actual_returns = np.diff(y) / y[:-1]

    # Interval constraint variable I_t
    # At trade step t (deciding on move t→t+1):
    #   check if actual price at t is inside the predicted band at t+1.
    #   If outside → model is misaligned with reality → skip trade (I_t = 0).
    if interval_lower is not None and interval_upper is not None:
        inside = ((y[:-1] >= interval_lower[1:]) &
                  (y[:-1] <= interval_upper[1:]))
        I_t = inside.astype(float)
    else:
        I_t = np.ones(n - 1)

    # Apply interval constraint to predicted return
    if scheme in [1, 2]:
        IR = pred_returns                # no constraint
    else:  # "1'" or "2'"
        IR = pred_returns * I_t          # zero out uncertain weeks

    # Generate signals
    portfolio_value = [INITIAL_CAPITAL]
    positions       = []
    n_trades        = 0

    for t in range(len(IR)):
        pr = IR[t]
        ar = actual_returns[t]

        if scheme in [1, "1'"]:
            # Trade on direction only
            if pr > 0:
                signal = 1    # long
            elif pr < 0:
                signal = -1   # short
            else:
                signal = 0    # hold
        else:
            # Only trade if predicted return exceeds transaction cost
            if pr > tc:
                signal = 1
            elif pr < -tc:
                signal = -1
            else:
                signal = 0

        positions.append(signal)

        if signal != 0:
            n_trades += 1
            # P&L = signal * actual return - transaction cost
            pnl = signal * ar - tc
        else:
            pnl = 0.0

        portfolio_value.append(portfolio_value[-1] * (1 + pnl))

    portfolio_value = np.array(portfolio_value)
    daily_returns   = np.diff(portfolio_value) / portfolio_value[:-1]

    # Metrics
    cumulative_return  = (portfolio_value[-1] - 1) * 100
    avg_daily_return   = np.mean(daily_returns) * 100

    # Maximum drawdown
    peak     = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - peak) / peak
    max_dd   = abs(drawdown.min()) * 100

    # Annualised Sharpe (weekly data, 52 weeks/year)
    if np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(52)
    else:
        sharpe = 0.0

    return {
        "Cumulative Return (%)":  round(cumulative_return, 4),
        "Avg Daily Return (%)":   round(avg_daily_return, 4),
        "Max Drawdown (%)":       round(max_dd, 4),
        "Sharpe (annualised)":    round(sharpe, 4),
        "N Transactions":         n_trades,
    }

# ── 5. TABLE 11 — Decomposition Model Trading Performance ─────
print("\nRunning trading simulations (decomposition models)...")

table11_rows = []
scheme_names = [1, "1'", 2, "2'"]

for scheme in scheme_names:
    for model_name, pred in decomp_preds.items():
        use_interval = "'" in str(scheme)
        lo = interval_lower if use_interval else None
        hi = interval_upper if use_interval else None
        result = run_trading_strategy(y_true, pred,
                                      scheme=scheme,
                                      interval_lower=lo,
                                      interval_upper=hi)
        result["Model"]  = model_name
        result["Scheme"] = f"Scheme {scheme}"
        table11_rows.append(result)

table11 = pd.DataFrame(table11_rows)
cols_order = ["Scheme", "Model", "Cumulative Return (%)",
              "Avg Daily Return (%)", "Max Drawdown (%)",
              "Sharpe (annualised)", "N Transactions"]
table11 = table11[cols_order]
table11.to_csv("table11_decomp_trading.csv", index=False)
print("Table 11 — Decomposition Model Trading:")
print(table11.to_string(index=False))
print("Saved: table11_decomp_trading.csv")

# ── 6. TABLE 12 — Single Model Trading Performance ────────────
print("\nRunning trading simulations (single models)...")

table12_rows = []
for scheme in scheme_names:
    for model_name, pred in single_preds.items():
        use_interval = "'" in str(scheme)
        lo = interval_lower if use_interval else None
        hi = interval_upper if use_interval else None
        result = run_trading_strategy(y_true, pred,
                                      scheme=scheme,
                                      interval_lower=lo,
                                      interval_upper=hi)
        result["Model"]  = model_name
        result["Scheme"] = f"Scheme {scheme}"
        table12_rows.append(result)

table12 = pd.DataFrame(table12_rows)
table12 = table12[cols_order]
table12.to_csv("table12_single_trading.csv", index=False)
print("Saved: table12_single_trading.csv")

# ── Interval constraint diagnostic (Proposed model) ───────────
print("\n" + "=" * 55)
print("INTERVAL CONSTRAINT DIAGNOSTIC — Proposed Model")
print("=" * 55)

prop_s1  = table11[(table11['Scheme'] == 'Scheme 1')  & (table11['Model'] == 'Proposed')].iloc[0]
prop_s1p = table11[(table11['Scheme'] == "Scheme 1'") & (table11['Model'] == 'Proposed')].iloc[0]

# Count weeks where constraint blocked a trade
# I_t = 0 when actual[t] is outside band[t+1]; that means signal would be non-zero but is zeroed out
y_arr   = np.array(y_true)
I_t_vec = ((y_arr[:-1] >= interval_lower[1:]) & (y_arr[:-1] <= interval_upper[1:])).astype(float)
prop_pred_returns = np.diff(np.array(proposed_pred)) / np.array(proposed_pred)[:-1]
# Weeks where constraint blocked: I_t == 0 AND signal would have been non-zero
would_trade  = (prop_pred_returns != 0)
blocked_mask = (I_t_vec == 0) & would_trade
n_blocked    = int(blocked_mask.sum())

print(f"  Scheme 1  — N transactions    : {int(prop_s1['N Transactions'])}")
print(f"  Scheme 1' — N transactions    : {int(prop_s1p['N Transactions'])}  (must be < Scheme 1)")
print(f"  Scheme 1  — Cumulative return : {prop_s1['Cumulative Return (%)']:.4f}%")
print(f"  Scheme 1' — Cumulative return : {prop_s1p['Cumulative Return (%)']:.4f}%")
print(f"  Weeks where constraint blocked a trade: {n_blocked}")

assert int(prop_s1p['N Transactions']) < int(prop_s1['N Transactions']), \
    "ERROR: Scheme 1' has >= transactions as Scheme 1 — constraint not working!"
print("  ✓ Scheme 1' has fewer transactions than Scheme 1")
print("=" * 55)

# ── Conditional DA — Proposed model Scheme 1' ─────────────────
from scipy.stats import binomtest

print("\n" + "=" * 60)
print("CONDITIONAL DA — Proposed Model, Scheme 1'")
print("=" * 60)

y_arr       = np.array(y_true)
p_arr       = np.array(proposed_pred)
act_ret     = np.diff(y_arr) / y_arr[:-1]          # actual weekly returns
pred_ret    = np.diff(p_arr) / p_arr[:-1]           # predicted weekly returns

# Interval constraint: I_t = 1 if actual[t] inside predicted band[t+1]
I_t_s1p = ((y_arr[:-1] >= interval_lower[1:]) &
            (y_arr[:-1] <= interval_upper[1:])).astype(float)

# Scheme 1' signals: direction of pred_ret, zeroed when I_t = 0
signals_s1p = np.where(I_t_s1p == 0, 0,
              np.where(pred_ret > 0,  1,
              np.where(pred_ret < 0, -1, 0)))

traded_mask   = signals_s1p != 0
correct_mask  = traded_mask & (signals_s1p * act_ret > 0)
wrong_mask    = traded_mask & (signals_s1p * act_ret <= 0)

n_total_weeks = len(act_ret)          # test weeks with a trade decision
n_traded      = int(traded_mask.sum())
n_correct     = int(correct_mask.sum())
n_wrong       = int(wrong_mask.sum())

cond_da   = n_correct / n_traded * 100 if n_traded > 0 else float('nan')
binom_p   = binomtest(n_correct, n_traded, p=0.5,
                      alternative='greater').pvalue if n_traded > 0 else float('nan')

# Weekly returns on correct and wrong trades (gross, before transaction cost)
# Return = signal * actual_return (positive = profit direction)
gross_ret_correct = (signals_s1p[correct_mask] * act_ret[correct_mask]) * 100
gross_ret_wrong   = (signals_s1p[wrong_mask]   * act_ret[wrong_mask])   * 100

avg_ret_correct = gross_ret_correct.mean() if n_correct > 0 else float('nan')
avg_ret_wrong   = gross_ret_wrong.mean()   if n_wrong   > 0 else float('nan')

print(f"  Total test weeks (trade decisions) : {n_total_weeks}")
print(f"  Weeks where trade was made         : {n_traded}  "
      f"({n_traded/n_total_weeks*100:.1f}% of test)")
print(f"  Of those — directionally correct   : {n_correct}")
print(f"  Of those — directionally wrong     : {n_wrong}")
print(f"  Conditional DA                     : {cond_da:.2f}%")
print(f"  Binomial p-value vs 50%            : {binom_p:.4f}"
      + ("  ***" if binom_p < 0.01 else "  **" if binom_p < 0.05
         else "  *" if binom_p < 0.10 else ""))
print(f"  Avg weekly return (correct trades) : +{avg_ret_correct:.3f}%")
print(f"  Avg weekly return (wrong trades)   :  {avg_ret_wrong:.3f}%")

# Profit Factor
profit_factor = (avg_ret_correct * n_correct) / (abs(avg_ret_wrong) * n_wrong) \
    if n_wrong > 0 and avg_ret_wrong != 0 else float('nan')

# Kelly Criterion
odds  = avg_ret_correct / abs(avg_ret_wrong) if avg_ret_wrong != 0 else float('nan')
da_frac = cond_da / 100
kelly = da_frac - (1 - da_frac) / odds if not np.isnan(odds) else float('nan')

print(f"  Profit factor                      :  {profit_factor:.4f}")
print(f"    = ({avg_ret_correct:.3f} × {n_correct}) / ({abs(avg_ret_wrong):.3f} × {n_wrong})")
print(f"  Kelly criterion (optimal bet size) :  {kelly*100:.2f}%")
print(f"    odds = {avg_ret_correct:.3f}/{abs(avg_ret_wrong):.3f} = {odds:.4f}")
print(f"    Kelly = {da_frac:.4f} - ({1-da_frac:.4f}/{odds:.4f}) = {kelly:.4f}")
print("=" * 60)

# ── Write / update paper_summary.json ─────────────────────────
import json, os

summary_path = "paper_summary.json"
summary = json.loads(open(summary_path).read()) if os.path.exists(summary_path) else {}

summary["scheme_1prime_proposed"] = {
    "total_test_weeks":        n_total_weeks,
    "weeks_traded":            n_traded,
    "weeks_traded_pct":        round(n_traded / n_total_weeks * 100, 2),
    "directionally_correct":   n_correct,
    "directionally_wrong":     n_wrong,
    "conditional_DA_pct":      round(cond_da, 4),
    "binomial_pvalue_vs50":    round(binom_p, 4),
    "avg_return_correct_pct":  round(avg_ret_correct, 4),
    "avg_return_wrong_pct":    round(avg_ret_wrong, 4),
    "profit_factor":           round(profit_factor, 4),
    "kelly_criterion_pct":     round(kelly * 100, 4),
    "odds_win_loss":           round(odds, 4),
    "cumulative_return_pct":   round(float(prop_s1p["Cumulative Return (%)"]), 4),
    "sharpe_annualised":       round(float(prop_s1p["Sharpe (annualised)"]), 4),
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {summary_path}")

# ── 7. FIG 11 — Interval Forecasting Results ──────────────────
print("\nPlotting Fig 11...")

fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(test_dates, interval_lower, interval_upper,
                alpha=0.3, color='#95a5a6', label='Predicted interval')
ax.plot(test_dates, proposed_pred, color='#27ae60',
        linewidth=1.2, label='Point prediction')
ax.plot(test_dates, y_true, color='#e74c3c',
        linewidth=1.0, linestyle='--', label='Actual', alpha=0.8)

ax.set_title('Interval Forecasting Results — Silver Price',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (INR/kg)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig11_interval_forecasts.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig11_interval_forecasts.png")

# ── 8. FIG 12 — Trading Strategy Illustration ─────────────────
print("Plotting Fig 12...")

# Show first 60 test observations for clarity (like paper)
n_show  = min(60, n_test)
show_dates = test_dates[:n_show]

# Find periods where point prediction falls outside interval (red circles)
outside_mask = ((proposed_pred[:n_show] < interval_lower[:n_show]) |
                (proposed_pred[:n_show] > interval_upper[:n_show]))

fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(show_dates,
                interval_lower[:n_show], interval_upper[:n_show],
                alpha=0.25, color='#27ae60',
                label='Predicted interval (lower/upper)')
ax.plot(show_dates, interval_upper[:n_show],
        color='#27ae60', linewidth=0.8, linestyle='-')
ax.plot(show_dates, interval_lower[:n_show],
        color='#27ae60', linewidth=0.8, linestyle='-')
ax.plot(show_dates, proposed_pred[:n_show],
        color='#8e44ad', linewidth=1.3, label='Point prediction')
ax.plot(show_dates, y_true[:n_show],
        color='#e74c3c', linewidth=1.0, linestyle='--',
        label='Actual', alpha=0.7)

# Mark periods of high uncertainty (point outside interval) with red circles
if outside_mask.any():
    ax.scatter(show_dates[outside_mask],
               proposed_pred[:n_show][outside_mask],
               color='red', s=60, zorder=5, label='High uncertainty (no trade)')

ax.set_title('Trading Strategy Illustration — Interval Constraint',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (INR/kg)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig12_trading_strategy_illustration.png", dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: fig12_trading_strategy_illustration.png")

# ── 9. FIG 13 — Trading Evaluation Bar Plots ──────────────────
print("Plotting Fig 13...")

all_trading = pd.concat([
    table11[table11['Model'].isin(decomp_preds.keys())],
    table12[table12['Model'].isin(single_preds.keys())]
])

scheme_colors = {
    'Scheme 1': '#5B9BD5',
    "Scheme 1'": '#E74C3C',
    'Scheme 2': '#70AD47',
    "Scheme 2'": '#C0504D',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
all_models = list(single_preds.keys()) + list(decomp_preds.keys())
x = np.arange(len(all_models))
width = 0.2

for ax_idx, metric in enumerate(['Cumulative Return (%)',
                                  'Sharpe (annualised)']):
    ax = axes[ax_idx]
    for s_idx, scheme in enumerate(['Scheme 1', "Scheme 1'",
                                     'Scheme 2', "Scheme 2'"]):
        vals = []
        for model in all_models:
            row = all_trading[(all_trading['Scheme'] == scheme) &
                              (all_trading['Model'] == model)]
            vals.append(float(row[metric].values[0]) if len(row) > 0 else 0)
        ax.bar(x + s_idx * width, vals, width,
               color=scheme_colors[scheme],
               label=scheme if ax_idx == 0 else '')

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    if ax_idx == 0:
        ax.legend(fontsize=8)

fig.suptitle('Trading Performance Across Models and Schemes',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig13_trading_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig13_trading_evaluation.png")

print("\n" + "=" * 55)
print("STEP 6 COMPLETE — All outputs produced!")
print("  table10_interval_errors.csv")
print("  table11_decomp_trading.csv")
print("  table12_single_trading.csv")
print("  fig11_interval_forecasts.png")
print("  fig12_trading_strategy_illustration.png")
print("  fig13_trading_evaluation.png")
print("=" * 55)
print("\n🎉 FULL PIPELINE COMPLETE!")
print("Upload all outputs to Claude for verification.")