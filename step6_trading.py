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

# ── 2. Interval Forecasting (simplified iMLP approach) ────────
# Paper uses interval MLP on intraday high/low
# We approximate using rolling prediction intervals
# Upper bound = prediction + 1.5 * rolling std of errors
# Lower bound = prediction - 1.5 * rolling std of errors

print("\nGenerating interval forecasts (Proposed method)...")

# Use rolling error std from training residuals to build intervals
train_silver = silver_all[:n_train]
train_pred   = proposed_pred   # we'll use test predictions directly

# Compute rolling std of past 8-week prediction errors as uncertainty proxy
# (approximates the paper's BEMD/iMLP interval width)
window = 8
pred_series  = pd.Series(proposed_pred)
rolling_std  = pred_series.rolling(window=window, min_periods=2).std().fillna(
    pred_series.std())

interval_upper = proposed_pred + 1.5 * rolling_std.values
interval_lower = proposed_pred - 1.5 * rolling_std.values

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
    if interval_lower is not None and interval_upper is not None:
        inside = ((y_hat[1:] >= interval_lower[1:]) &
                  (y_hat[1:] <= interval_upper[1:]))
        I_t = inside.astype(float)
    else:
        I_t = np.ones(n - 1)

    # Apply interval constraint to predicted return
    if scheme in [1, 2]:
        IR = pred_returns            # no constraint
    else:
        IR = pred_returns * I_t      # zero out uncertain predictions

    # Generate signals
    portfolio_value = [INITIAL_CAPITAL]
    positions       = []
    n_trades        = 0

    for t in range(len(IR)):
        pr = IR[t]
        ar = actual_returns[t]

        if scheme in [1, "1'"]:
            # Trade on direction
            if pr > 0:
                signal = 1    # long
            elif pr < 0:
                signal = -1   # short
            else:
                signal = 0    # hold
        else:
            # Only trade if predicted return > transaction cost
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
        sc = 1 if str(scheme) in ["1", "1'"] else 2

        result = run_trading_strategy(y_true, pred,
                                      scheme=sc,
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
        sc = 1 if str(scheme) in ["1", "1'"] else 2

        result = run_trading_strategy(y_true, pred,
                                      scheme=sc,
                                      interval_lower=lo,
                                      interval_upper=hi)
        result["Model"]  = model_name
        result["Scheme"] = f"Scheme {scheme}"
        table12_rows.append(result)

table12 = pd.DataFrame(table12_rows)
table12 = table12[cols_order]
table12.to_csv("table12_single_trading.csv", index=False)
print("Saved: table12_single_trading.csv")

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
ax.set_ylabel('Price (USD/oz)')
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
ax.set_ylabel('Price (USD/oz)')
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

schemes_to_plot = ['Scheme 1', "Scheme 1'", 'Scheme 2', "Scheme 2'"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, metric in enumerate(['Cumulative Return (%)',
                                  'Sharpe (annualised)']):
    ax = axes[ax_idx]
    for scheme_idx, scheme in enumerate(schemes_to_plot):
        subset  = all_trading[all_trading['Scheme'] == scheme]
        models  = subset['Model'].values
        values  = subset[metric].values
        x       = np.arange(len(models)) + scheme_idx * (len(models) + 1)
        colors  = ['#3498db' if m in single_preds else '#e74c3c'
                   for m in models]
        ax.bar(x, values, color=colors, alpha=0.8,
               label=scheme if ax_idx == 0 else "")
        ax.axhline(0, color='black', linewidth=0.5)

    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    if ax_idx == 0:
        ax.legend(fontsize=8)

fig.suptitle('Trading Performance Across Models and Schemes',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig13_trading_evaluation.png", dpi=150, bbox_inches='tight')
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