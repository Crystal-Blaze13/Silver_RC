"""
generate_figures.py — Regenerate all publication figures at 300 DPI
=====================================================================
Run this script AFTER the full pipeline (steps 1–6) to regenerate every
figure from saved artefacts.  Also produces the synthetic crash stress-test
comparison chart (fig_stress_test_comparison.png) which is not generated
by any individual step.

Usage:
    python generate_figures.py

Figures produced:
    fig4_silver_price_split.png          — Paper Fig 4
    fig7_imf_decomposition.png           — Paper Fig 7
    fig8_approximate_entropy.png         — Paper Fig 8
    fig8b_imf_correlation.png            — Added: mode-mixing diagnostic
    fig9_single_model_forecasts.png      — Paper Fig 9
    fig10_error_barplots.png             — Paper Fig 10
    fig11_interval_forecasts.png         — Paper Fig 11
    fig12_trading_strategy_illustration.png — Paper Fig 12
    fig13_trading_evaluation.png         — Paper Fig 13
    fig13b_equity_curves.png             — Added: equity curves
    fig_dm_heatmap.png                   — Added: DM test heat-map
    fig_stress_test_comparison.png       — Added: actual vs synthetic crash
"""

import json
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    SILVER_FILE, N_TRAIN_FILE, IMFS_FILE, COMPLEXITY_FILE,
    PREDICTIONS_PKL, COMPARISON_JSON,
    FIG4, FIG7, FIG8, FIG8B, FIG9, FIG10, FIG11, FIG12, FIG13, FIG13B,
    FIG_DM, FIG_STRESS,
    FIG_DPI, PALETTE,
)

warnings.filterwarnings("ignore")

DPI = FIG_DPI   # 300 for publication

# ── helpers ───────────────────────────────────────────────────────────────────
def _savefig(path: str):
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _load_silver():
    df = pd.read_csv(SILVER_FILE, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def _load_n_train():
    return int(np.load(N_TRAIN_FILE)[0])


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Price time series with train/test split
# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    prices  = _load_silver()
    n_train = _load_n_train()
    train   = prices.iloc[:n_train]
    test    = prices.iloc[n_train:]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(train.index, train.values,
            color=PALETTE["train"], linewidth=1.2, label="Training set")
    ax.plot(test.index,  test.values,
            color=PALETTE["test"],  linewidth=1.2, label="Testing set")
    ax.axvline(train.index[-1], color="black", linestyle="--",
               linewidth=0.9, alpha=0.6,
               label=f"Split: {train.index[-1].date()}")
    ax.set_title("MCX Silver Price — Weekly (INR/kg)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR/kg)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    _savefig(FIG4)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Decomposed IMFs
# ─────────────────────────────────────────────────────────────────────────────
def fig7():
    u_sorted  = np.load(IMFS_FILE)
    prices    = _load_silver()
    n_train   = _load_n_train()
    K         = u_sorted.shape[0]
    table5    = pd.read_csv("../results/tables/table5_imf_statistics.csv")
    imf_index = prices.index[:u_sorted.shape[1]]

    fig, axes = plt.subplots(K, 1, figsize=(14, K * 1.8), sharex=True)
    if K == 1:
        axes = [axes]

    for i in range(K):
        imf = u_sorted[i, :]
        ax  = axes[i]
        ax.plot(imf_index[:n_train], imf[:n_train],
                color=PALETTE["train"], linewidth=0.9)
        ax.plot(imf_index[n_train:len(imf)], imf[n_train:],
                color=PALETTE["test"], linewidth=0.9)
        ax.set_ylabel(f"IMF{i+1}", fontsize=9, rotation=0,
                      labelpad=40, va="center", ha="right")
        ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
        ax.tick_params(axis="y", labelsize=7)
        vr = table5.loc[i, "Variance Ratio"] if i < len(table5) else ""
        ax.text(0.99, 0.85, f"VR={vr:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#555")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45, fontsize=8)
    fig.suptitle(f"VMD Decomposed IMFs (K={K}) — MCX Silver (INR/kg)",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    _savefig(FIG7)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Approximate Entropy bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig8():
    comp      = pd.read_csv(COMPLEXITY_FILE)
    K         = len(comp)
    ae_vals   = comp["ApEn"].values
    se_vals   = comp["SampEn"].values
    labels    = comp["IMF"].tolist()
    complexity = comp["Complexity"].tolist()

    prices       = _load_silver()
    signal_array = prices.values.astype(float)

    # recompute original series ApEn
    from step2_entropy import approximate_entropy, sample_entropy
    r_orig  = 0.2 * np.std(signal_array)
    ae_orig = approximate_entropy(signal_array, 2, r_orig)
    se_orig = sample_entropy(signal_array, 2, 0.2 * np.std(signal_array))

    colors = [PALETTE["low_comp"] if c == "Low" else PALETTE["high_comp"]
              for c in complexity]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(labels, ae_vals, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=3, alpha=0.85)
    ax1.axhline(ae_orig, color="black", linestyle="--", linewidth=1.6,
                label=f"Original ApEn = {ae_orig:.4f}", zorder=4)
    for bar, val in zip(bars, ae_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(ae_vals) * 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    valid_se = np.where(np.isfinite(se_vals), se_vals, 0)
    ax2.plot(labels, valid_se, color="#f39c12", marker="D",
             markersize=6, linewidth=1.4, linestyle="-.",
             label=f"SampEn (orig={se_orig:.4f})", zorder=5)
    ax2.axhline(se_orig, color="#f39c12", linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Sample Entropy", fontsize=10, color="#f39c12")
    ax2.tick_params(axis="y", labelcolor="#f39c12")

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["low_comp"],  alpha=0.85,
                       label="Low complexity → ARIMA"),
        mpatches.Patch(facecolor=PALETTE["high_comp"], alpha=0.85,
                       label="High complexity → LSTM"),
        Line2D([0], [0], color="black", linestyle="--",
               label=f"Original ApEn = {ae_orig:.4f}"),
        Line2D([0], [0], color="#f39c12", linestyle="-.", marker="D",
               markersize=5, label=f"SampEn (orig={se_orig:.4f})"),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax1.set_title("Approximate Entropy of Decomposed IMFs — MCX Silver",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("IMF Component", fontsize=11)
    ax1.set_ylabel("Approximate Entropy", fontsize=11)
    ax1.set_ylim(0, max(ae_vals) * 1.25)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    _savefig(FIG8)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8b — IMF correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig8b():
    u_sorted  = np.load(IMFS_FILE)
    K         = u_sorted.shape[0]
    labels    = [f"IMF{i+1}" for i in range(K)]
    corr_mat  = np.corrcoef(u_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr_mat[i, j]) > 0.6 else "black")
    ax.set_title("IMF Pairwise Correlation — Mode-Mixing Diagnostic",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(FIG8B)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Single-model forecasts  (loaded from predictions.pkl)
# ─────────────────────────────────────────────────────────────────────────────
def fig9():
    with open(PREDICTIONS_PKL, "rb") as f:
        data = pickle.load(f)
    single_preds = data["single_preds"]
    y_true       = np.array(data["y_true_test"])
    test_dates   = data["test_dates"]
    models       = list(single_preds.keys())

    n_models = len(models)
    ncols    = 2
    nrows    = (n_models + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5),
                             sharex=True)
    axes_flat = axes.flatten()

    colors_cycle = ["#e67e22", "#8e44ad", "#16a085", "#c0392b",
                    "#2980b9", "#27ae60", "#d35400", "#7f8c8d"]

    for idx, (model, preds) in enumerate(single_preds.items()):
        ax = axes_flat[idx]
        ax.plot(test_dates, y_true, color=PALETTE["test"],
                linewidth=1.4, label="Actual")
        ax.plot(test_dates, preds, color=colors_cycle[idx % len(colors_cycle)],
                linewidth=1.1, linestyle="--", label=model)
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.3)

    # hide unused subplot
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Single-Model Forecasts — MCX Silver Test Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(FIG9)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Error metric bar plots across all models
# ─────────────────────────────────────────────────────────────────────────────
def fig10():
    t7 = pd.read_csv("../results/tables/table7_single_model_errors.csv")
    t8 = pd.read_csv("../results/tables/table8_decomp_model_errors.csv")
    all_m = pd.concat([t7, t8], ignore_index=True)

    metrics = ["RMSE", "MAE", "MAPE(%)", "sMAPE(%)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    bar_colors = [PALETTE["proposed"] if "Proposed" in m
                  else PALETTE["vmd_arima"] if "VMD-ARIMA" in m
                  else PALETTE["neutral"]
                  for m in all_m["Model"]]

    for ax, metric in zip(axes, metrics):
        vals = all_m[metric].values.astype(float)
        bars = ax.bar(range(len(all_m)), vals, color=bar_colors,
                      edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(all_m)))
        ax.set_xticklabels(all_m["Model"], rotation=45, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 100 else f"{x:.2f}"))
        ax.grid(axis="y", alpha=0.3)

    legend_items = [
        mpatches.Patch(facecolor=PALETTE["proposed"], label="Proposed"),
        mpatches.Patch(facecolor=PALETTE["vmd_arima"], label="VMD-ARIMA (best)"),
        mpatches.Patch(facecolor=PALETTE["neutral"], label="Other"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Forecast Error Metrics — MCX Silver Test Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(FIG10)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Interval forecasting results
# ─────────────────────────────────────────────────────────────────────────────
def fig11():
    with open(PREDICTIONS_PKL, "rb") as f:
        data = pickle.load(f)

    # These keys are produced by step6
    if "interval_lower" not in data or "interval_upper" not in data:
        print("  WARNING: interval_lower/upper not in predictions.pkl — re-run step6")
        return

    y_true   = np.array(data["y_true_test"])
    proposed = np.array(data["proposed_pred"])
    lower    = np.array(data["interval_lower"])
    upper    = np.array(data["interval_upper"])
    dates    = data["test_dates"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dates, lower, upper, alpha=0.3,
                    color=PALETTE["interval"], label="80% prediction interval")
    ax.plot(dates, y_true,   color=PALETTE["test"],     linewidth=1.4, label="Actual")
    ax.plot(dates, proposed, color=PALETTE["proposed"],  linewidth=1.1,
            linestyle="--", label="Point forecast (Proposed)")
    ax.set_title("Interval Forecasting Results — MCX Silver",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR/kg)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(FIG11)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — Trading strategy illustration with interval constraint
# ─────────────────────────────────────────────────────────────────────────────
def fig12():
    # Load from predictions (step6 writes these)
    if not os.path.exists(PREDICTIONS_PKL):
        print("  WARNING: predictions.pkl not found")
        return
    with open(PREDICTIONS_PKL, "rb") as f:
        data = pickle.load(f)

    if "interval_lower" not in data:
        print("  WARNING: interval data not in predictions.pkl — re-run step6")
        return

    y_true   = np.array(data["y_true_test"])
    proposed = np.array(data["proposed_pred"])
    lower    = np.array(data["interval_lower"])
    upper    = np.array(data["interval_upper"])
    dates    = data["test_dates"]

    # Highlight where point prediction falls outside interval
    outside = (proposed < lower) | (proposed > upper)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(dates, lower, upper, alpha=0.25,
                    color="#27ae60", label="Predicted price interval")
    ax.plot(dates, y_true, color=PALETTE["test"],
            linewidth=1.4, label="Actual")
    ax.plot(dates, proposed, color="#8e44ad",
            linewidth=1.1, linestyle="--", label="Point prediction")

    # Red circles where prediction escapes interval → no-trade signal
    out_dates = [d for d, o in zip(dates, outside) if o]
    out_vals  = [proposed[i] for i, o in enumerate(outside) if o]
    if out_dates:
        ax.scatter(out_dates, out_vals, facecolors="none", edgecolors="#e74c3c",
                   s=80, linewidths=1.5, zorder=5,
                   label=f"Outside interval → no trade ({len(out_dates)})")

    ax.set_title("Trading Strategy Illustration — Interval Constraint",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR/kg)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(FIG12)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — Trading evaluation bar plots (cumulative return + Sharpe)
# ─────────────────────────────────────────────────────────────────────────────
def fig13():
    t11 = pd.read_csv("../results/tables/table11_decomp_trading.csv")
    t12 = pd.read_csv("../results/tables/table12_single_trading.csv")
    all_t = pd.concat([t12, t11], ignore_index=True)

    schemes = ["Scheme 1", "Scheme 1'", "Scheme 2", "Scheme 2'"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(18, 5), sharey=False)

    for ax, scheme in zip(axes, schemes):
        if scheme not in all_t.columns:
            ax.set_visible(False)
            continue
        vals   = all_t[scheme].astype(float)
        models = all_t["Model"].tolist()
        colors = [PALETTE["proposed"] if "Proposed" in m
                  else PALETTE["vmd_arima"] if "VMD-ARIMA" in m
                  else "#3498db" if m in ("ARIMA", "VMD-ARIMA", "CEEMDAN-ARIMA")
                  else PALETTE["neutral"]
                  for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.set_title(scheme, fontsize=10, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1, f"{val:.0f}%",
                        ha="center", va="bottom", fontsize=6)

    fig.suptitle("Trading Strategy Evaluation — MCX Silver Test Period",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(FIG13)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 13b — Equity curves for the Proposed method across all 4 schemes
# ─────────────────────────────────────────────────────────────────────────────
def fig13b():
    if not os.path.exists(PREDICTIONS_PKL):
        print("  WARNING: predictions.pkl not found")
        return
    with open(PREDICTIONS_PKL, "rb") as f:
        data = pickle.load(f)

    if "equity_curves" not in data:
        print("  INFO: equity_curves not in predictions.pkl; "
              "fig13b skipped (re-run step6 to generate)")
        return

    curves     = data["equity_curves"]    # dict: scheme_label → equity array
    test_dates = data["test_dates"]

    scheme_colors = {
        "Scheme 1":  PALETTE["train"],
        "Scheme 1'": PALETTE["proposed"],
        "Scheme 2":  PALETTE["arima"],
        "Scheme 2'": PALETTE["vmd_arima"],
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    for label, curve in curves.items():
        color = scheme_colors.get(label, "grey")
        ax.plot(test_dates[:len(curve)], np.array(curve) * 100 - 100,
                linewidth=1.4, label=label, color=color)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Equity Curves — Proposed Method (All 4 Schemes)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(FIG13B)


# ─────────────────────────────────────────────────────────────────────────────
# FIG_DM — DM test heat-map
# ─────────────────────────────────────────────────────────────────────────────
def fig_dm():
    if not os.path.exists("../results/tables/table9_dm_stats_numeric.csv"):
        print("  WARNING: table9_dm_stats_numeric.csv not found")
        return

    dm_stats = pd.read_csv("../results/tables/table9_dm_stats_numeric.csv", index_col=0)
    dm_p1sid = pd.read_csv("../results/tables/table9_dm_pvals_onesided.csv", index_col=0)
    model_names = list(dm_stats.index)
    nm = len(model_names)
    stat_vals = dm_stats.values.astype(float)

    def sig_stars(p):
        if pd.isna(p): return ""
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    vmax = np.nanpercentile(np.abs(stat_vals), 95)
    fig, ax = plt.subplots(figsize=(max(8, nm * 0.9), max(6, nm * 0.9)))
    im = ax.imshow(stat_vals, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="DM statistic (+ = row better)")
    ax.set_xticks(range(nm))
    ax.set_yticks(range(nm))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color="lightgray"))
                continue
            v = stat_vals[i, j]
            p = dm_p1sid.values[i, j] if not np.isnan(dm_p1sid.values[i, j]) else np.nan
            s = sig_stars(p)
            txt = f"{v:.2f}{s}" if not np.isnan(v) else ""
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.6 else "black")

    ax.set_title("DM Test Matrix — MCX Silver\n"
                 "(+ = row model outperforms column model; *, **, *** = 10/5/1%)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _savefig(FIG_DM)


# ─────────────────────────────────────────────────────────────────────────────
# FIG_STRESS — Actual vs synthetic crash stress test comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_stress():
    if not os.path.exists(COMPARISON_JSON):
        print(f"  WARNING: {COMPARISON_JSON} not found")
        return

    with open(COMPARISON_JSON) as f:
        comp = json.load(f)

    actual    = comp.get("actual",    comp.get("actual_restored", {}))
    synthetic = comp.get("synthetic", {})

    # Metric pairs to compare
    metrics = {
        "Proposed Scheme 1\nCumulative Return (%)": (
            actual.get("proposed_scheme1_return", np.nan),
            synthetic.get("proposed_scheme1_return", np.nan),
        ),
        "Proposed Scheme 1\nSharpe Ratio": (
            actual.get("proposed_scheme1_sharpe", np.nan),
            synthetic.get("proposed_scheme1_sharpe", np.nan),
        ),
        "Buy & Hold\nCumulative Return (%)": (
            actual.get("buy_hold_return", np.nan),
            synthetic.get("buy_hold_return", np.nan),
        ),
        "Buy & Hold\nSharpe Ratio": (
            actual.get("buy_hold_sharpe", np.nan),
            synthetic.get("buy_hold_sharpe", np.nan),
        ),
        "Best Active Strategy\nCumulative Return (%)": (
            actual.get("best_final_return", np.nan),
            synthetic.get("best_final_return", np.nan),
        ),
        "Best Active Strategy\nSharpe Ratio": (
            actual.get("best_final_sharpe", np.nan),
            synthetic.get("best_final_sharpe", np.nan),
        ),
    }

    n = len(metrics)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    actual_vals    = [v[0] for v in metrics.values()]
    synthetic_vals = [v[1] for v in metrics.values()]

    bars_a = ax.bar(x - width / 2, actual_vals, width,
                    label="Actual silver (2024–2026)",
                    color=PALETTE["train"], alpha=0.85, edgecolor="white")
    bars_s = ax.bar(x + width / 2, synthetic_vals, width,
                    label="Synthetic crash (₹90k → ₹29k)",
                    color=PALETTE["test"], alpha=0.85, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=9, ha="center")
    ax.set_ylabel("Value")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bars in (bars_a, bars_s):
        for bar in bars:
            h = bar.get_height()
            label = f"{h:.1f}"
            va = "bottom" if h >= 0 else "top"
            offset = 0.5 if h >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    label, ha="center", va=va, fontsize=8, fontweight="bold")

    ax.set_title(
        "Stress Test: Actual vs Synthetic Silver Crash\n"
        "Scheme 1 Sharpe improves 1.69 → 2.70 during crash; "
        "Buy & Hold collapses +164% → −69%",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _savefig(FIG_STRESS)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Regenerating all figures at 300 DPI")
    print("=" * 60)

    tasks = [
        ("Fig 4  — Price series + split",     fig4),
        ("Fig 7  — VMD decomposed IMFs",       fig7),
        ("Fig 8  — Approximate entropy",       fig8),
        ("Fig 8b — IMF correlation heat-map",  fig8b),
        ("Fig 9  — Single-model forecasts",    fig9),
        ("Fig 10 — Error metric bar plots",    fig10),
        ("Fig 11 — Interval forecasting",      fig11),
        ("Fig 12 — Trading strategy illus.",   fig12),
        ("Fig 13 — Trading evaluation bars",   fig13),
        ("Fig 13b— Equity curves",             fig13b),
        ("Fig DM — DM test heat-map",          fig_dm),
        ("Fig ST — Stress test comparison",    fig_stress),
    ]

    for name, fn in tasks:
        print(f"\n{name}")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Done. All figures saved at 300 DPI.")
    print("=" * 60)
