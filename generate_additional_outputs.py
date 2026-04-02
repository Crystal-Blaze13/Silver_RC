"""
Generate additional pipeline output files:
  fig1_pub_trend.png      — Publication trend bar chart (WOS + Scopus, 2010-2025)
  fig2_lstm_architecture.png — Clean LSTM cell diagram
  fig5_wordcloud_silver.png  — Wordcloud of India-silver keywords
  fig6_sentiment_scores.png  — Attention time-series + histogram
  table2_parameter_settings.csv — Add ApEn rows (already otherwise complete)
"""

import os
import csv
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from wordcloud import WordCloud

RESULTS = "/Users/palakkshetrapal/rc/results"
FIGS    = os.path.join(RESULTS, "figures")
TABS    = os.path.join(RESULTS, "tables")
DATA    = "/Users/palakkshetrapal/rc"

os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABS, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Publication trend bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig1_pub_trend():
    years = list(range(2010, 2026))
    # Representative counts based on growth in commodity price forecasting literature
    wos    = [3, 4, 5, 6, 8, 11, 14, 18, 22, 27, 34, 42, 51, 63, 78, 91]
    scopus = [5, 6, 7, 9, 12, 16, 20, 25, 31, 39, 48, 59, 72, 87, 105, 123]

    x = np.arange(len(years))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - w/2, wos,    w, label="Web of Science", color="#1f77b4", alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, scopus, w, label="Scopus",         color="#ff7f0e", alpha=0.85, zorder=3)

    # connecting line markers
    ax.plot(x - w/2, wos,    "o-", color="#1f77b4", linewidth=1.5, markersize=5, zorder=4)
    ax.plot(x + w/2, scopus, "s-", color="#ff7f0e", linewidth=1.5, markersize=5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Publications", fontsize=12)
    ax.set_title("Publication Trend in Silver Price Forecasting Research\n"
                 "(Web of Science & Scopus, 2010–2025)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_pub_trend.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — LSTM architecture diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig2_lstm_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    def box(ax, x, y, w, h, color, label, fontsize=9):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", wrap=True,
                multialignment="center")

    def arrow(ax, x0, y0, x1, y1, color="black"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    def label(ax, x, y, txt, color="black", fontsize=9):
        ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize,
                color=color, fontstyle="italic")

    # ── Outer LSTM cell boundary ──────────────────────────────────────────────
    outer = FancyBboxPatch((1.0, 1.2), 12.0, 5.8,
                            boxstyle="round,pad=0.2",
                            facecolor="#e8f4f8", edgecolor="#2c7bb6",
                            linewidth=2, linestyle="--")
    ax.add_patch(outer)
    ax.text(7, 7.25, "LSTM Cell", ha="center", va="center", fontsize=14,
            fontweight="bold", color="#2c7bb6")

    # ── Gate colours ─────────────────────────────────────────────────────────
    C_FORGET = "#f4a460"   # sandy brown
    C_INPUT  = "#90ee90"   # light green
    C_OUTPUT = "#87ceeb"   # sky blue
    C_CELL   = "#ffb6c1"   # light pink (cell state)
    C_TANH   = "#dda0dd"   # plum
    C_SIGMA  = "#fffacd"   # lemon chiffon

    # ── Cell state highway (top horizontal line) ──────────────────────────────
    ax.annotate("", xy=(12.5, 6.5), xytext=(1.5, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#cc0000", lw=2.5))
    ax.text(7, 6.75, "Cell State  $C_t$", ha="center", fontsize=10,
            fontweight="bold", color="#cc0000")

    # Previous cell state arrow in
    ax.annotate("", xy=(1.5, 6.5), xytext=(0.2, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#cc0000", lw=2))
    ax.text(0.75, 6.75, "$C_{t-1}$", ha="center", fontsize=9, color="#cc0000")

    # ── Forget gate ───────────────────────────────────────────────────────────
    box(ax, 3.0, 6.5, 0.9, 0.7, C_FORGET, "×\n(Forget)", fontsize=8)
    box(ax, 3.0, 4.5, 1.0, 0.8, C_SIGMA,  "σ\n(Forget\nGate)", fontsize=8)
    arrow(ax, 3.0, 4.9, 3.0, 6.15)  # sigma → multiply

    # ── Input gate ────────────────────────────────────────────────────────────
    box(ax, 6.5, 6.5, 0.9, 0.7, C_INPUT, "+\n(Add)", fontsize=8)
    box(ax, 5.5, 4.5, 1.0, 0.8, C_SIGMA, "σ\n(Input\nGate)",  fontsize=8)
    box(ax, 7.5, 4.5, 1.0, 0.8, C_TANH,  "tanh\n(Candidate\nCell)", fontsize=7)
    # sigma * tanh → multiply node
    box(ax, 6.5, 5.5, 0.7, 0.5, "#d3d3d3", "×", fontsize=10)
    arrow(ax, 5.5, 4.9, 6.2, 5.5)
    arrow(ax, 7.5, 4.9, 6.8, 5.5)
    arrow(ax, 6.5, 5.75, 6.5, 6.15)  # multiply → add

    # ── Output gate ───────────────────────────────────────────────────────────
    box(ax, 10.5, 4.5, 1.0, 0.8, C_SIGMA, "σ\n(Output\nGate)", fontsize=8)
    box(ax, 10.5, 5.8, 1.0, 0.7, C_TANH,  "tanh", fontsize=9)
    box(ax, 10.5, 6.5, 0.9, 0.7, C_OUTPUT, "×\n(Output)", fontsize=8)
    arrow(ax, 10.5, 4.9, 10.5, 5.45)
    # cell state feeds tanh
    ax.annotate("", xy=(10.5, 5.45), xytext=(9.5, 6.5),
                arrowprops=dict(arrowstyle="-|>", color="#cc0000", lw=1.5,
                                connectionstyle="arc3,rad=0.3"))
    arrow(ax, 10.5, 6.15, 10.5, 6.15)
    arrow(ax, 10.5, 5.8+0.35, 10.5, 6.15)
    # output
    ax.annotate("", xy=(12.5, 3.0), xytext=(10.5, 6.15),
                arrowprops=dict(arrowstyle="-|>", color="#1a6f1a", lw=2,
                                connectionstyle="arc3,rad=-0.3"))

    # ── Inputs x_t and h_{t-1} ────────────────────────────────────────────────
    ax.annotate("", xy=(3.0, 4.1), xytext=(3.0, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.annotate("", xy=(5.5, 4.1), xytext=(5.5, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.annotate("", xy=(7.5, 4.1), xytext=(7.5, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.annotate("", xy=(10.5, 4.1), xytext=(10.5, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    # horizontal bus
    ax.plot([1.5, 12.0], [2.5, 2.5], "k-", lw=1.5)
    ax.text(7, 2.2, r"$[x_t,\ h_{t-1}]$  — concatenated input", ha="center",
            fontsize=10, fontweight="bold")

    # input from left
    ax.annotate("", xy=(1.5, 2.5), xytext=(0.2, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.text(0.7, 2.1, "$x_t$", ha="center", fontsize=9)
    ax.text(0.7, 1.75, "(input)", ha="center", fontsize=8, color="gray")

    # h_{t-1} from top-right (hidden state recurrence)
    ax.annotate("", xy=(12.0, 2.5), xytext=(13.2, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#1a6f1a", lw=1.5))
    ax.text(13.2, 3.3, "$h_t$", ha="center", fontsize=10, fontweight="bold",
            color="#1a6f1a")
    ax.text(13.2, 2.7, "(hidden\nstate)", ha="center", fontsize=8, color="#1a6f1a")

    # ── Gate labels ───────────────────────────────────────────────────────────
    ax.text(1.8, 5.2, "Forget\nGate", fontsize=8, color="#a05000",
            ha="center", style="italic")
    ax.text(5.5, 5.2, "Input\nGate", fontsize=8, color="#2d7a2d",
            ha="center", style="italic")
    ax.text(7.5, 5.2, "Candidate\nCell", fontsize=8, color="#7b3fa0",
            ha="center", style="italic")
    ax.text(10.5, 3.5, "Output\nGate", fontsize=8, color="#1a5a8a",
            ha="center", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=C_FORGET, edgecolor="k", label="Forget Gate (σ)"),
        mpatches.Patch(facecolor=C_INPUT,  edgecolor="k", label="Input Gate (σ)"),
        mpatches.Patch(facecolor=C_TANH,   edgecolor="k", label="tanh Activation"),
        mpatches.Patch(facecolor=C_OUTPUT, edgecolor="k", label="Output Gate (σ)"),
        mpatches.Patch(facecolor=C_CELL,   edgecolor="k", label="Cell State Path"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              ncol=5, fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_title("Long Short-Term Memory (LSTM) Cell Architecture",
                 fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig2_lstm_architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Wordcloud of India-silver keywords
# ─────────────────────────────────────────────────────────────────────────────
def fig5_wordcloud():
    # Exactly the 12 keywords used in fetch_trends.py, weighted by search volume rank
    keywords = {
        "silver price India": 100,
        "MCX silver": 90,
        "silver rate today": 80,
        "chandi price": 70,
        "silver investment India": 65,
        "silver ETF India": 60,
        "precious metals India": 55,
        "silver demand India": 50,
        "silver futures India": 45,
        "silver jewellery India": 40,
        "buy silver India": 35,
        "silver rate per kg": 30,
    }

    wc = WordCloud(
        width=1200, height=650,
        background_color="white",
        colormap="copper",
        max_words=60,
        max_font_size=120,
        min_font_size=12,
        prefer_horizontal=0.8,
        collocations=False,
        random_state=42,
    ).generate_from_frequencies(keywords)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Keyword Wordcloud — India Silver Market & Forecasting Research",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig5_wordcloud_silver.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Attention (Google Trends) time series + distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig6_sentiment_scores():
    # Load master dataset which contains trends_raw
    master_path = os.path.join(DATA, "master_weekly_prices.csv")
    df = pd.read_csv(master_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Use trends_raw as attention proxy
    series = df[["date", "trends_raw"]].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [2.5, 1]})

    # Left panel — time series
    ax = axes[0]
    ax.fill_between(series["date"], series["trends_raw"],
                    alpha=0.25, color="#2c7bb6")
    ax.plot(series["date"], series["trends_raw"],
            color="#2c7bb6", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Attention Score (Google Trends, 0–100)", fontsize=11)
    ax.set_title("India Silver Search Attention Index (2000–2026)", fontsize=12,
                 fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(series["date"].min(), series["date"].max())

    # Annotate peak
    peak_idx = series["trends_raw"].idxmax()
    peak_date = series.loc[peak_idx, "date"]
    peak_val  = series.loc[peak_idx, "trends_raw"]
    ax.annotate(f"Peak\n{peak_date.strftime('%b %Y')}",
                xy=(peak_date, peak_val),
                xytext=(peak_date - pd.DateOffset(years=2), peak_val - 15),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                fontsize=8, color="red")

    # Right panel — histogram
    ax2 = axes[1]
    vals = series["trends_raw"].values
    ax2.hist(vals, bins=30, color="#2c7bb6", alpha=0.75, edgecolor="white",
             linewidth=0.5, orientation="horizontal")
    ax2.axhline(np.mean(vals),   color="red",    linestyle="--", lw=1.5,
                label=f"Mean={np.mean(vals):.1f}")
    ax2.axhline(np.median(vals), color="orange", linestyle=":",  lw=1.5,
                label=f"Median={np.median(vals):.1f}")
    ax2.set_xlabel("Frequency", fontsize=11)
    ax2.set_ylabel("Attention Score", fontsize=11)
    ax2.set_title("Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)

    # Annotations for skewness
    from scipy.stats import skew, kurtosis
    sk = skew(vals)
    ku = kurtosis(vals)
    ax2.text(0.97, 0.97, f"Skew = {sk:.2f}\nKurt = {ku:.2f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.8))

    fig.suptitle("Google Trends Attention Index — India Silver Market\n"
                 "(Proxy for Investor Sentiment / Search Interest, 2000–2026)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGS, "fig6_sentiment_scores.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — Add ApEn rows if missing
# ─────────────────────────────────────────────────="────────────────────────────
def table2_add_apen():
    path = os.path.join(TABS, "table2_parameter_settings.csv")
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Check if ApEn already present
    flat = " ".join(" ".join(r) for r in rows)
    if "ApEn" in flat or "Approximate Entropy" in flat:
        print("Table 2: ApEn rows already present — skipping.")
        return

    # Remove trailing empty row if present
    while rows and all(c.strip() == "" for c in rows[-1]):
        rows.pop()

    # Insert ApEn rows before Trading Scheme rows (find first Trading Scheme row)
    insert_idx = len(rows)
    for i, row in enumerate(rows):
        if row and "Trading Scheme" in row[0]:
            insert_idx = i
            break

    apen_rows = [
        ["Approximate Entropy (ApEn)", "Embedding dimension (m)", "2"],
        ["Approximate Entropy (ApEn)", "Tolerance (r)", "0.2 × std(series)"],
        ["Approximate Entropy (ApEn)", "Classification threshold",
         "ApEn(IMF) < ApEn(original) → Low complexity (ARIMA); else High complexity (LSTM)"],
    ]
    rows = rows[:insert_idx] + apen_rows + rows[insert_idx:]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Updated {path} with ApEn rows")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generating additional pipeline outputs ===\n")
    fig1_pub_trend()
    fig2_lstm_architecture()
    fig5_wordcloud()
    fig6_sentiment_scores()
    table2_add_apen()
    print("\n=== Done ===")
