"""
Generate Bloomberg Data Reference PDF
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

OUT = "bloomberg_data_reference.pdf"

C_HEADER  = "#1a3a5c"
C_SUBHEAD = "#2e6da4"
C_ROW_A   = "#eef4fb"
C_ROW_B   = "#ffffff"
C_CODE_BG = "#f4f4f4"
C_BORDER  = "#c0cfe0"
C_TEXT    = "#1a1a2e"
C_NOTE    = "#555577"

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"


def add_page(pdf, fig_func):
    fig = fig_func()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── PAGE 1: Overview table ─────────────────────────────────────
def page1():
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=C_HEADER, zorder=1))
    ax.text(0.5, 0.95, "Bloomberg Data Reference",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", fontfamily=SANS, zorder=2)
    ax.text(0.5, 0.905, "Silver Price Forecasting Pipeline — Indian Market (2000–2026)",
            ha="center", va="center", fontsize=10, color="#aac8e8",
            fontfamily=SANS, zorder=2)

    ax.text(0.06, 0.855, "Data Sources Overview",
            fontsize=13, fontweight="bold", color=C_HEADER, fontfamily=SANS)
    ax.axhline(0.845, xmin=0.06, xmax=0.94, color=C_SUBHEAD, linewidth=1.2)

    col_labels = ["Variable", "Pipeline Column", "Bloomberg Ticker", "Unit"]
    col_x      = [0.06, 0.28, 0.52, 0.78]
    row_h      = 0.052
    y_start    = 0.825

    rows = [
        ("Silver Futures",  "silver",      "SI1 Comdty",     "USD / oz"),
        ("Gold Futures",    "gold",        "GC1 Comdty",     "USD / oz"),
        ("Brent Crude",     "brent",       "CO1 Comdty",     "USD / bbl"),
        ("USD / INR Spot",  "usdinr",      "USDINR Curncy",  "INR per USD"),
        ("Nifty 50 Index",  "nifty50",     "NIFTY Index",    "Index pts"),
        ("CBOE VIX",        "vix",         "VIX Index",      "Index"),
        ("Google Trends",   "trends_raw",  "— (pytrends)",   "0–100 norm."),
    ]

    y = y_start
    ax.add_patch(plt.Rectangle((0.05, y - 0.005), 0.90, row_h,
                               color=C_SUBHEAD, zorder=1))
    for label, x in zip(col_labels, col_x):
        ax.text(x + 0.01, y + row_h / 2 - 0.003, label,
                fontsize=8.5, fontweight="bold", color="white",
                va="center", fontfamily=SANS, zorder=2)
    y -= row_h

    for i, (var, col, bbg, unit) in enumerate(rows):
        bg = C_ROW_A if i % 2 == 0 else C_ROW_B
        ax.add_patch(plt.Rectangle((0.05, y - 0.005), 0.90, row_h,
                                   color=bg, zorder=1,
                                   linewidth=0.5, edgecolor=C_BORDER))
        for text, x in zip([var, col, bbg, unit], col_x):
            fam = MONO if 0.25 < x < 0.78 else SANS
            ax.text(x + 0.01, y + row_h / 2 - 0.003, text,
                    fontsize=8, color=C_TEXT, va="center",
                    fontfamily=fam, zorder=2)
        y -= row_h

    y_note = y - 0.035
    ax.add_patch(FancyBboxPatch((0.05, y_note - 0.075), 0.90, 0.085,
                                boxstyle="round,pad=0.01",
                                facecolor="#fff8e7", edgecolor="#e0c060",
                                linewidth=1, zorder=1))
    ax.text(0.095, y_note - 0.005, "Notes",
            fontsize=8.5, fontweight="bold", color="#8a6000", fontfamily=SANS)
    notes = [
        "• All series: weekly frequency (W-SUN), last price, 2000-01-01 to 2026-03-20.",
        "• Per=W resamples to weekly.  Fill=P forward-fills holidays — matches pipeline ffill(limit=2).",
        "• Google Trends has no Bloomberg equivalent; continue using pytrends / manual CSV.",
        "• After export, rename columns to: date, silver, gold, brent, usdinr, nifty50, vix  before build_master.py.",
    ]
    for j, note in enumerate(notes):
        ax.text(0.07, y_note - 0.022 - j * 0.015, note,
                fontsize=7.5, color=C_NOTE, fontfamily=SANS, va="top")

    ax.axhline(0.035, xmin=0.06, xmax=0.94, color=C_BORDER, linewidth=0.8)
    ax.text(0.5, 0.018, "Page 1 of 2  ·  Silver Price Forecasting Pipeline — Indian Market",
            ha="center", fontsize=7, color=C_NOTE, fontfamily=SANS)

    return fig


# ── PAGE 2: BDH formulas ───────────────────────────────────────
def page2():
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=C_HEADER, zorder=1))
    ax.text(0.5, 0.95, "Bloomberg Excel (BDH) Formulas",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white", fontfamily=SANS, zorder=2)
    ax.text(0.5, 0.905, "Silver Price Forecasting Pipeline — Indian Market (2000–2026)",
            ha="center", va="center", fontsize=10, color="#aac8e8",
            fontfamily=SANS, zorder=2)

    instruments = [
        {
            "title":   "Silver  —  COMEX Front-Month Futures",
            "ticker":  "SI1 Comdty",
            "formula": '=BDH("SI1 Comdty","PX_LAST","1/1/2000","3/20/2026","Per=W","Fill=P","Days=A")',
        },
        {
            "title":   "Gold  —  COMEX Front-Month Futures",
            "ticker":  "GC1 Comdty",
            "formula": '=BDH("GC1 Comdty","PX_LAST","1/1/2000","3/20/2026","Per=W","Fill=P","Days=A")',
        },
        {
            "title":   "Brent Crude  —  Front-Month Futures",
            "ticker":  "CO1 Comdty",
            "formula": '=BDH("CO1 Comdty","PX_LAST","1/1/2000","3/20/2026","Per=W","Fill=P","Days=A")',
        },
        {
            "title":   "USD / INR  —  Spot Rate",
            "ticker":  "USDINR Curncy",
            "formula": '=BDH("USDINR Curncy","PX_LAST","1/1/2000","3/20/2026","Per=W","Fill=P","Days=A")',
        },
        {
            "title":   "Nifty 50  —  NSE Benchmark Index",
            "ticker":  "NIFTY Index",
            "formula": '=BDH("NIFTY Index","PX_LAST","1/1/2000","3/20/2026","Per=W","Fill=P","Days=A")',
        },
        {
            "title":   "VIX  —  CBOE Volatility Index",
            "ticker":  "VIX Index",
            "formula": '=BDH("VIX Index","PX_LAST","1/1/2000","3/20/2026","Per=W","Fill=P","Days=A")',
        },
    ]

    y       = 0.845
    block_h = 0.107
    gap     = 0.012

    for inst in instruments:
        ax.add_patch(FancyBboxPatch((0.05, y - block_h + 0.005), 0.90, block_h,
                                    boxstyle="round,pad=0.008",
                                    facecolor=C_ROW_A, edgecolor=C_BORDER,
                                    linewidth=0.8, zorder=1))
        ax.text(0.07, y - 0.012, inst["title"],
                fontsize=9, fontweight="bold", color=C_HEADER,
                fontfamily=SANS, va="top", zorder=2)
        ax.add_patch(FancyBboxPatch((0.07, y - 0.042), 0.20, 0.022,
                                    boxstyle="round,pad=0.004",
                                    facecolor=C_SUBHEAD, edgecolor="none", zorder=2))
        ax.text(0.17, y - 0.031, inst["ticker"],
                fontsize=8, color="white", fontfamily=MONO,
                ha="center", va="center", zorder=3)
        ax.add_patch(FancyBboxPatch((0.07, y - 0.088), 0.86, 0.030,
                                    boxstyle="round,pad=0.004",
                                    facecolor=C_CODE_BG, edgecolor=C_BORDER,
                                    linewidth=0.6, zorder=2))
        ax.text(0.085, y - 0.073, inst["formula"],
                fontsize=7.2, color=C_TEXT, fontfamily=MONO,
                va="center", zorder=3)
        y -= (block_h + gap)

    y_leg = y - 0.018
    ax.add_patch(FancyBboxPatch((0.05, y_leg - 0.065), 0.90, 0.072,
                                boxstyle="round,pad=0.01",
                                facecolor="#f0f0f8", edgecolor="#9090c0",
                                linewidth=0.8, zorder=1))
    ax.text(0.095, y_leg - 0.006, "BDH Parameter Reference",
            fontsize=8.5, fontweight="bold", color=C_HEADER, fontfamily=SANS)
    params = [
        ('"PX_LAST"', "Last / closing price field"),
        ('"Per=W"',   "Weekly periodicity — matches pipeline W-SUN resampling"),
        ('"Fill=P"',  "Forward-fill missing values (holidays / non-trading days)"),
        ('"Days=A"',  "All calendar days before weekly roll (drop for trading days only)"),
    ]
    for j, (param, desc) in enumerate(params):
        ax.text(0.08,  y_leg - 0.022 - j * 0.013, param,
                fontsize=7.5, color=C_SUBHEAD, fontfamily=MONO, va="top")
        ax.text(0.22, y_leg - 0.022 - j * 0.013, f"—  {desc}",
                fontsize=7.5, color=C_NOTE, fontfamily=SANS, va="top")

    ax.axhline(0.035, xmin=0.06, xmax=0.94, color=C_BORDER, linewidth=0.8)
    ax.text(0.5, 0.018, "Page 2 of 2  ·  Silver Price Forecasting Pipeline — Indian Market",
            ha="center", fontsize=7, color=C_NOTE, fontfamily=SANS)

    return fig


with PdfPages(OUT) as pdf:
    add_page(pdf, page1)
    add_page(pdf, page2)

print(f"Saved: {OUT}")
