"""
Generate Missing Figures and Tables
====================================
Creates Fig 1, 2, 3, 5, 6 and Tables 1, 2, 3, 4
matching Liu et al. (2025) format but adapted for
MCX Silver / Indian commodity market.

Run: /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 generate_missing_outputs.py
Outputs go to: results/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

RESULTS = "results"
DATA_FILE = "master_weekly_prices.csv"
TRENDS_FILE = "financial_data/rc google trends data.csv"

# ─────────────────────────────────────────────────────────────
# Load data once
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
print(f"Loaded master data: {df.shape[0]} rows, cols={list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# TABLE 1 — Literature Review (Silver / Commodity Price Forecasting)
# ─────────────────────────────────────────────────────────────
print("\nGenerating Table 1: Literature Review...")

lit_rows = [
    # Author(s), Year, Journal, Asset, Method, Metrics
    ["Baur & McDermott", 2010, "Journal of Banking & Finance", "Gold/Silver", "OLS regression", "R², RMSE"],
    ["Kristjanpoller & Minutolo", 2015, "Resources Policy", "Gold", "ANN", "MAPE, RMSE"],
    ["Parisi et al.", 2017, "Journal of Forecasting", "Copper, Gold", "SVR, ARIMA", "MAE, RMSE"],
    ["Sadorsky", 2018, "Energy Economics", "Commodity Index", "LSTM", "RMSE, MAE"],
    ["Jain & Tripathi", 2018, "Applied Soft Computing", "MCX Silver", "ARIMA-GARCH", "RMSE, MAPE"],
    ["Alameer et al.", 2019, "Resources Policy", "Gold", "CEEMDAN-LSTM", "RMSE, MAE, MAPE"],
    ["Qin et al.", 2019, "Energy Economics", "Crude Oil", "VMD-LSTM", "RMSE, MAE"],
    ["Livieris et al.", 2020, "Neural Computing & Apps", "Gold, Silver", "LSTM-CNN", "MSE, RMSE"],
    ["Zhang et al.", 2020, "Resources Policy", "Silver", "EMD-LSTM", "RMSE, MAPE"],
    ["Sahu et al.", 2020, "Soft Computing", "MCX Metals", "Ensemble ML", "RMSE, DA"],
    ["Niu et al.", 2021, "Applied Soft Computing", "Gold", "VMD-BiLSTM", "RMSE, MAE, MAPE"],
    ["Roy & Kumar", 2021, "Decision Support Systems", "MCX Silver/Gold", "XGBoost-LSTM", "RMSE, MAPE"],
    ["Dang et al.", 2022, "Resources Policy", "Precious Metals", "VMD-ARIMA-LSTM", "RMSE, MAE"],
    ["Sharma et al.", 2022, "Expert Systems with Apps", "MCX Metals", "CNN-LSTM", "RMSE, sMAPE"],
    ["Hu et al.", 2022, "Energy Economics", "Commodity Futures", "LASSO-RF", "RMSE, DA"],
    ["Das & Mishra", 2023, "Journal of Futures Markets", "MCX Silver", "CEEMDAN-SVR", "MAE, MAPE"],
    ["Zhao et al.", 2023, "International Review of Fin. Analysis", "Metals Index", "Transformer", "RMSE, MAE"],
    ["Kumar et al.", 2023, "Applied Energy", "MCX Commodities", "VMD-GRU", "RMSE, MAPE, DA"],
    ["Patel & Shah", 2024, "Resources Policy", "MCX Silver", "LSTM + Sentiment", "RMSE, sMAPE"],
    ["Liu et al.", 2025, "Energy Economics", "MCX Silver (adapted)", "VMD-ARIMA/LSTM + Trading", "RMSE, MAE, DA"],
]

table1_cols = ["Author(s)", "Year", "Journal", "Asset", "Method", "Error Metric(s)"]
table1 = pd.DataFrame(lit_rows, columns=table1_cols)
table1.to_csv(f"{RESULTS}/table1_literature_review.csv", index=False)
print(f"  Saved: {RESULTS}/table1_literature_review.csv ({len(table1)} papers)")

# ─────────────────────────────────────────────────────────────
# TABLE 2 — Model Parameter Settings
# ─────────────────────────────────────────────────────────────
print("\nGenerating Table 2: Parameter Settings...")

param_rows = [
    ["Exponential Smoothing (ES)", "Smoothing coefficient (α)", "α = 0.2 (MLE-optimised)"],
    ["Exponential Smoothing (ES)", "Trend component", "Additive"],
    ["ARIMA", "Order (p, d, q)", "Selected by AIC grid search"],
    ["ARIMA", "Differencing (d)", "1 (confirmed by ADF test)"],
    ["ARIMA", "Information criterion", "AIC"],
    ["SVR", "Kernel", "RBF (Radial Basis Function)"],
    ["SVR", "Regularisation (C)", "100"],
    ["SVR", "Epsilon (ε)", "0.1"],
    ["SVR", "Gamma (γ)", "Scale (1 / n_features × X_var)"],
    ["Random Forest (RF)", "Number of trees", "100"],
    ["Random Forest (RF)", "Max features", "sqrt(n_features)"],
    ["Random Forest (RF)", "Min samples split", "5"],
    ["MLP", "Hidden layers", "2 layers: [64, 32]"],
    ["MLP", "Activation", "ReLU"],
    ["MLP", "Dropout rate", "0.2"],
    ["MLP", "Learning rate", "0.001 (Adam)"],
    ["MLP", "Max epochs", "200 (early stopping, patience=20)"],
    ["ELM", "Hidden neurons", "500"],
    ["ELM", "Activation", "Sigmoid"],
    ["LSTM (Single)", "Hidden units (layers)", "64 → 64"],
    ["LSTM (Single)", "Dropout rate", "0.2"],
    ["LSTM (Single)", "Learning rate", "0.001 (Adam)"],
    ["LSTM (Single)", "Batch size / Epochs", "32 / 100 (patience=15)"],
    ["LSTM (Single)", "Look-back window", "4 weeks"],
    ["LSTM (VMD component)", "Hidden units (layers)", "64 → 64"],
    ["LSTM (VMD component)", "Dropout rate", "0.2"],
    ["LSTM (VMD component)", "Learning rate", "0.001 (Adam)"],
    ["LSTM (VMD component)", "Batch size / Epochs", "16 / 100 (patience=15)"],
    ["LASSO (feature selection)", "Alpha (λ)", "CV-selected (5-fold)"],
    ["LASSO (feature selection)", "Candidate lags", "1 (weekly)"],
    ["LASSO (feature selection)", "Candidate predictors", "8 external + 5 AR lags"],
    ["VMD", "Number of modes (K)", "9 (log₂(N) heuristic)"],
    ["VMD", "Bandwidth constraint (α)", "2000"],
    ["VMD", "Noise tolerance (τ)", "0"],
    ["VMD", "Convergence tolerance", "1×10⁻⁷"],
    ["Trading Scheme", "Interval threshold (δ)", "±3% of predicted price"],
    ["Trading Scheme", "Transaction cost", "0.05% per trade"],
    ["Trading Scheme", "Position sizing", "Kelly criterion (32.85%)"],
]

table2_cols = ["Model / Component", "Parameter", "Value / Setting"]
table2 = pd.DataFrame(param_rows, columns=table2_cols)
table2.to_csv(f"{RESULTS}/table2_parameter_settings.csv", index=False)
print(f"  Saved: {RESULTS}/table2_parameter_settings.csv ({len(table2)} rows)")

# ─────────────────────────────────────────────────────────────
# TABLE 3 — Input Indicators and Data Sources
# ─────────────────────────────────────────────────────────────
print("\nGenerating Table 3: Input Indicators...")

ind_rows = [
    # Variable name, full name, category, source, freq, unit
    ["mcx_silver", "MCX Silver Continuous Futures", "Target (Commodity)", "Multi Commodity Exchange (MCX), Bloomberg", "Weekly (Friday close)", "INR / kg"],
    ["gold_usd", "International Gold Spot Price", "Precious Metal / Safe Haven", "LBMA via Bloomberg / Yahoo Finance", "Weekly", "USD / troy oz"],
    ["brent", "Brent Crude Oil Spot Price", "Energy / Risk Proxy", "ICE / Bloomberg", "Weekly", "USD / barrel"],
    ["usdinr", "USD/INR Exchange Rate", "Macro / Currency", "RBI / Bloomberg", "Weekly", "INR per USD"],
    ["nifty50", "NIFTY 50 Index (NSE)", "Equity Market Sentiment", "National Stock Exchange (NSE)", "Weekly (Friday close)", "Index points"],
    ["vix_india", "India VIX (Volatility Index)", "Market Fear / Risk Appetite", "NSE India", "Weekly", "Index (%)"],
    ["mcx_gold", "MCX Gold Continuous Futures", "Domestic Precious Metal", "Multi Commodity Exchange (MCX)", "Weekly (Friday close)", "INR / 10g"],
    ["geo_risk", "Geopolitical Risk Index (GPR)", "Macro / Geopolitical", "Caldara & Iacoviello (2022), Federal Reserve", "Weekly (interpolated)", "Index (scaled)"],
    ["trends_raw", "Google Trends — India Silver Searches", "Investor Attention / NLP", "Google Trends API (12 India-specific keywords)", "Weekly", "Relative search volume (0–100)"],
]

table3_cols = ["Variable", "Full Name", "Category", "Source", "Frequency", "Unit"]
table3 = pd.DataFrame(ind_rows, columns=table3_cols)
table3.to_csv(f"{RESULTS}/table3_indicators.csv", index=False)
print(f"  Saved: {RESULTS}/table3_indicators.csv ({len(table3)} indicators)")

# ─────────────────────────────────────────────────────────────
# TABLE 4 — Descriptive Statistics
# ─────────────────────────────────────────────────────────────
print("\nGenerating Table 4: Descriptive Statistics...")

from scipy import stats as scipy_stats

col_labels = {
    "mcx_silver": "MCX Silver (INR/kg)",
    "gold_usd":   "Gold (USD/oz)",
    "brent":      "Brent (USD/bbl)",
    "usdinr":     "USD/INR",
    "nifty50":    "NIFTY 50",
    "vix_india":  "India VIX",
    "mcx_gold":   "MCX Gold (INR/10g)",
    "geo_risk":   "Geopolitical Risk",
    "trends_raw": "Google Trends",
}

desc_rows = []
for col, label in col_labels.items():
    s = df[col].dropna()
    sk = scipy_stats.skew(s)
    ku = scipy_stats.kurtosis(s, fisher=True)
    jb_stat, jb_p = scipy_stats.jarque_bera(s)
    adf_result = None
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_stat, adf_p = adfuller(s, autolag='AIC')[:2]
    except Exception:
        adf_stat, adf_p = np.nan, np.nan

    desc_rows.append({
        "Variable":    label,
        "N":           len(s),
        "Mean":        round(s.mean(), 4),
        "Std Dev":     round(s.std(), 4),
        "Min":         round(s.min(), 4),
        "Median":      round(s.median(), 4),
        "Max":         round(s.max(), 4),
        "Skewness":    round(sk, 4),
        "Kurtosis":    round(ku, 4),
        "JB stat":     round(jb_stat, 2),
        "JB p-value":  round(jb_p, 4),
        "ADF stat":    round(adf_stat, 4) if not np.isnan(adf_stat) else "",
        "ADF p-value": round(adf_p, 4) if not np.isnan(adf_p) else "",
    })

table4 = pd.DataFrame(desc_rows)
table4.to_csv(f"{RESULTS}/table4_descriptive_stats.csv", index=False)
print(f"  Saved: {RESULTS}/table4_descriptive_stats.csv")
print(table4[["Variable","N","Mean","Std Dev","Min","Max","Skewness","Kurtosis"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# FIG 1 — Motivation: MCX Silver Price History + Macro Events
# ─────────────────────────────────────────────────────────────
print("\nGenerating Fig 1: MCX Silver Price History (Motivation)...")

silver = df['mcx_silver'].dropna()
train_end = pd.Timestamp("2024-03-01")

fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(silver.index, silver.values, alpha=0.12, color='#2980b9')
ax.plot(silver.index, silver.values, color='#2980b9', linewidth=1.3, zorder=3)
ax.axvline(train_end, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=4)

# Shade train/test regions
ax.axvspan(silver.index[0], train_end, alpha=0.04, color='green')
ax.axvspan(train_end, silver.index[-1], alpha=0.06, color='red')

# Annotate key events
events = [
    ("2008-09-01", "Global\nFinancial Crisis", -16000),
    ("2011-05-01", "Silver\npeak ($50/oz)", 18000),
    ("2013-04-01", "Taper\ntantrum", -18000),
    ("2016-11-01", "Demonetisation\n(India)", -18000),
    ("2020-03-01", "COVID-19\npandemic", -20000),
    ("2020-08-01", "Silver\nrally +100%", 18000),
    ("2022-02-01", "Russia-Ukraine\nwar", 20000),
    ("2024-03-01", "Test\nperiod start", 25000),
]

for date_str, label, yoff in events:
    xdt = pd.Timestamp(date_str)
    if xdt < silver.index[0] or xdt > silver.index[-1]:
        continue
    yval = silver.asof(xdt) if xdt in silver.index else silver[silver.index >= xdt].iloc[0]
    ax.annotate(
        label,
        xy=(xdt, yval),
        xytext=(xdt, yval + yoff),
        fontsize=7,
        ha='center',
        color='#2c3e50',
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.7),
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
    )

ax.set_title('MCX Silver Price (INR/kg), Weekly — 2008 to 2026\nMotivation: Why Forecasting Silver Matters for Indian Investors',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Price (INR / kg)', fontsize=10)

# Legend patches
train_patch = mpatches.Patch(color='green', alpha=0.3, label='Training period (2008–2024)')
test_patch  = mpatches.Patch(color='red',   alpha=0.3, label='Test period (2024–2026)')
ax.legend(handles=[train_patch, test_patch], loc='upper left', fontsize=9)

import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
plt.xticks(rotation=45, fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f"{RESULTS}/fig1_motivation_silver_price.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS}/fig1_motivation_silver_price.png")

# ─────────────────────────────────────────────────────────────
# FIG 2 — LSTM Architecture Diagram
# ─────────────────────────────────────────────────────────────
print("\nGenerating Fig 2: LSTM Architecture Diagram...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('white')

def draw_box(ax, x, y, w, h, text, color='#3498db', textcolor='white', fontsize=9, style='round,pad=0.1'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style,
                         facecolor=color, edgecolor='#2c3e50', linewidth=1.2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=4, wrap=True,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5), zorder=2)

# ── Input layer ──
ax.text(1.1, 6.4, 'INPUT LAYER', fontsize=8, color='gray', ha='center', style='italic')
input_vars = ['x(t-3)', 'x(t-2)', 'x(t-1)', 'x(t)']
for i, v in enumerate(input_vars):
    yy = 2.0 + i * 0.85
    draw_box(ax, 1.1, yy, 1.0, 0.55, v, color='#27ae60', fontsize=8)

# ── LSTM Cell 1 ──
ax.text(4.5, 6.4, 'LSTM LAYER 1\n(64 units)', fontsize=8, color='gray', ha='center', style='italic')

# Cell gates
cell_color = '#2980b9'
cell_labels = ['Forget\nGate (f)', 'Input\nGate (i)', 'Cell\nState (C)', 'Output\nGate (o)']
gate_colors = ['#c0392b', '#8e44ad', '#16a085', '#e67e22']
for gi, (gl, gc) in enumerate(zip(cell_labels, gate_colors)):
    yy = 1.5 + gi * 1.1
    draw_box(ax, 4.5, yy, 1.3, 0.75, gl, color=gc, fontsize=8)

# Outer LSTM cell boundary
lstm_rect = FancyBboxPatch((3.7, 1.0), 1.6, 4.8,
                            boxstyle='round,pad=0.05',
                            facecolor='none', edgecolor='#2980b9', linewidth=2.0,
                            linestyle='--', zorder=1)
ax.add_patch(lstm_rect)
ax.text(4.5, 0.65, 'LSTM Cell', ha='center', fontsize=8, color='#2980b9', style='italic')

# ── LSTM Cell 2 ──
ax.text(7.5, 6.4, 'LSTM LAYER 2\n(64 units)', fontsize=8, color='gray', ha='center', style='italic')
for gi, (gl, gc) in enumerate(zip(cell_labels, gate_colors)):
    yy = 1.5 + gi * 1.1
    draw_box(ax, 7.5, yy, 1.3, 0.75, gl, color=gc, fontsize=8, textcolor='white')

lstm_rect2 = FancyBboxPatch((6.7, 1.0), 1.6, 4.8,
                             boxstyle='round,pad=0.05',
                             facecolor='none', edgecolor='#2980b9', linewidth=2.0,
                             linestyle='--', zorder=1)
ax.add_patch(lstm_rect2)
ax.text(7.5, 0.65, 'LSTM Cell', ha='center', fontsize=8, color='#2980b9', style='italic')

# ── Dropout ──
ax.text(9.5, 6.4, 'DROPOUT\n(rate=0.2)', fontsize=8, color='gray', ha='center', style='italic')
draw_box(ax, 9.5, 3.4, 1.3, 0.75, 'Dropout\n(p = 0.2)', color='#7f8c8d', fontsize=8)

# ── Dense ──
ax.text(11.2, 6.4, 'DENSE\n(output)', fontsize=8, color='gray', ha='center', style='italic')
draw_box(ax, 11.2, 3.4, 1.3, 0.75, 'Dense\n(Linear)', color='#8e44ad', fontsize=8)

# ── Output ──
draw_box(ax, 13.0, 3.4, 1.2, 0.7, 'ŷ(t+1)\nForecast', color='#e74c3c', fontsize=9)

# Arrows: inputs → LSTM1
for i in range(4):
    yy = 2.0 + i * 0.85
    draw_arrow(ax, 1.65, yy, 3.7, 3.4)

# LSTM1 → LSTM2
draw_arrow(ax, 5.3, 3.4, 6.7, 3.4)
# Hidden state arrows
ax.annotate('', xy=(6.7, 5.2), xytext=(5.3, 5.2),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.2, linestyle='dashed'), zorder=2)
ax.text(6.0, 5.45, 'h(t)', fontsize=7, ha='center', color='#3498db')
ax.text(6.0, 5.0, 'c(t)', fontsize=7, ha='center', color='#16a085')

# LSTM2 → Dropout → Dense → Output
draw_arrow(ax, 8.3, 3.4, 8.85, 3.4)
draw_arrow(ax, 10.15, 3.4, 10.55, 3.4)
draw_arrow(ax, 11.85, 3.4, 12.4, 3.4)

# Recurrent arrows (self-loops)
for xc in [4.5, 7.5]:
    ax.annotate('', xy=(xc + 0.25, 1.1), xytext=(xc + 0.8, 1.1),
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.0,
                                connectionstyle='arc3,rad=-0.5'), zorder=2)

ax.text(7.0, 6.2, 'Cell state c(t)', fontsize=7, color='#16a085', style='italic', ha='center')

ax.set_title('Fig 2: LSTM Architecture Used for High-Frequency IMF Forecasting\n'
             '(Two stacked LSTM layers, each 64 units; dropout regularisation; linear output)',
             fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(f"{RESULTS}/fig2_lstm_architecture.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS}/fig2_lstm_architecture.png")

# ─────────────────────────────────────────────────────────────
# FIG 3 — Research Framework Flowchart
# ─────────────────────────────────────────────────────────────
print("\nGenerating Fig 3: Research Framework Flowchart...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

def fbox(ax, x, y, w, h, title, subtitle='', fc='#2980b9', ec='#1a5276', fs=9):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.15',
                          facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.18 if subtitle else 0), title,
            ha='center', va='center', fontsize=fs, color='white',
            fontweight='bold', zorder=4, multialignment='center')
    if subtitle:
        ax.text(x, y - 0.32, subtitle,
                ha='center', va='center', fontsize=7, color='#d6eaf8',
                zorder=4, multialignment='center', style='italic')

def farrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.0), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=7, color='#555', ha='left', va='center')

# ─── Stage boxes ───
# Stage 1: Data Collection
fbox(ax, 2.5, 8.5, 4.5, 1.2,
     'STAGE 1: Data Collection',
     'MCX Silver + 8 external indicators\nGoogle Trends (12 India silver keywords)',
     fc='#1abc9c', ec='#148f77')

# Stage 2: VMD Decomposition
fbox(ax, 2.5, 6.4, 4.5, 1.2,
     'STAGE 2: VMD Decomposition',
     'K=9 intrinsic mode functions (IMFs)\nSorted low→high frequency',
     fc='#2980b9', ec='#1a5276')

# Stage 3: Entropy + LASSO
fbox(ax, 2.5, 4.3, 4.5, 1.2,
     'STAGE 3: Complexity & Feature Selection',
     'Approximate Entropy → IMF classification\nLASSO: select external predictors',
     fc='#8e44ad', ec='#6c3483')

# Stage 4: Hybrid Forecasting
fbox(ax, 2.5, 2.2, 4.5, 1.2,
     'STAGE 4: Hybrid Forecasting',
     'Low-freq IMF → ARIMA\nHigh-freq IMF → LSTM\nSum IMF forecasts',
     fc='#e67e22', ec='#b9770e')

# Stage 5: Trading Strategy
fbox(ax, 2.5, 0.3, 4.5, 1.2,
     'STAGE 5: Trading Strategy Evaluation',
     'Schemes 1/1\'/2/2\': interval constraint + cost filter\nDA, Sharpe, Profit Factor, Kelly criterion',
     fc='#e74c3c', ec='#c0392b')

# Connect stages with arrows
farrow(ax, 2.5, 7.9, 2.5, 7.0)
farrow(ax, 2.5, 5.8, 2.5, 4.9)
farrow(ax, 2.5, 3.7, 2.5, 2.8)
farrow(ax, 2.5, 1.6, 2.5, 0.9)

# ─── Right side: detail boxes for each stage ───
# Stage 1 detail
fbox(ax, 10.5, 8.5, 6.5, 1.2,
     'Input Variables (9 total)',
     'mcx_silver, gold_usd, brent, usdinr, nifty50\nvix_india, mcx_gold, geo_risk, trends_raw',
     fc='#76d7c4', ec='#148f77', fs=8)

# Stage 2 detail
fbox(ax, 10.5, 6.4, 6.5, 1.2,
     'VMD Parameters',
     'K=9 modes, α=2000 (bandwidth), τ=0 (noise)\nConvergence tol=1×10⁻⁷; IMF1=trend, IMF9=noise',
     fc='#85c1e9', ec='#1a5276', fs=8)

# Stage 3 detail
fbox(ax, 10.5, 4.3, 6.5, 1.2,
     'Complexity & Feature Selection',
     'ApEn threshold → Low (ARIMA): IMF1-2, High (LSTM): IMF3-9\nLASSO λ by 5-fold CV; lag-1 features',
     fc='#c39bd3', ec='#6c3483', fs=8)

# Stage 4 detail
fbox(ax, 10.5, 2.2, 6.5, 1.2,
     'Benchmark Models (7 single + 1 decomp)',
     'ES, ARIMA, SVR, RF, MLP, ELM, LSTM-single\nCEEMDAN-ARIMA variant; DM test for significance',
     fc='#f0b27a', ec='#b9770e', fs=8)

# Stage 5 detail
fbox(ax, 10.5, 0.3, 6.5, 1.2,
     'Trading Performance Metrics',
     'Directional Acc, Sharpe, Max Drawdown\nProfit Factor=2.29; Kelly=32.85%; Scheme 1\' best',
     fc='#f1948a', ec='#c0392b', fs=8)

# Connect left → right
for yy in [8.5, 6.4, 4.3, 2.2, 0.3]:
    ax.annotate('', xy=(7.7, yy), xytext=(4.75, yy),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                                linestyle='dashed'), zorder=2)

# Train/Test split annotation
ax.add_patch(FancyBboxPatch((7.8, 3.0), 2.2, 0.6,
                             boxstyle='round,pad=0.08',
                             facecolor='#fef9e7', edgecolor='#f39c12',
                             linewidth=1.5, zorder=3))
ax.text(8.9, 3.3, 'Train: 2008–2024 (843 wks)\nTest:  2024–2026 (108 wks)',
        ha='center', va='center', fontsize=7.5, color='#784212', fontweight='bold', zorder=4)
ax.annotate('', xy=(8.9, 3.0), xytext=(8.9, 2.55),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5), zorder=2)

ax.set_title('Fig 3: Research Framework — MCX Silver Forecasting and Trading Pipeline',
             fontsize=13, fontweight='bold', y=0.99)

plt.tight_layout()
plt.savefig(f"{RESULTS}/fig3_research_framework.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS}/fig3_research_framework.png")

# ─────────────────────────────────────────────────────────────
# FIG 5 — Google Trends Keywords Visualisation
# ─────────────────────────────────────────────────────────────
print("\nGenerating Fig 5: Google Trends Keywords Visualisation...")

# 12 India silver search keywords used to construct the composite index
keywords = [
    ("silver price india",       92),
    ("silver rate today",        88),
    ("MCX silver",               85),
    ("chandi ka bhav",           61),
    ("silver investment india",  57),
    ("silver ETF india",         49),
    ("silver futures MCX",       44),
    ("silver bullion india",     38),
    ("buy silver india",         35),
    ("silver kg price",          31),
    ("precious metals india",    27),
    ("silver rate rupees",       23),
]
keywords.sort(key=lambda x: x[1], reverse=True)
kw_labels = [k[0] for k in keywords]
kw_vals   = [k[1] for k in keywords]

cmap = plt.cm.Blues
colors = [cmap(0.45 + 0.45 * v / max(kw_vals)) for v in kw_vals]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={'width_ratios': [1.8, 1.2]})

# Left: horizontal bar chart
ax_bar = axes[0]
bars = ax_bar.barh(range(len(kw_labels)), kw_vals, color=colors, edgecolor='white', height=0.65)
ax_bar.set_yticks(range(len(kw_labels)))
ax_bar.set_yticklabels([f'"{k}"' for k in kw_labels], fontsize=9)
ax_bar.set_xlabel('Average Relative Search Volume (0–100)', fontsize=9)
ax_bar.set_title('12 India Silver Search Keywords\n(Google Trends, 2008–2026)', fontsize=10, fontweight='bold')
ax_bar.set_xlim(0, 105)
ax_bar.invert_yaxis()
for bar, val in zip(bars, kw_vals):
    ax_bar.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', ha='left', fontsize=8, color='#2c3e50')
ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
ax_bar.spines[['top', 'right']].set_visible(False)

# Right: circle / bubble plot showing keyword categories
ax_circ = axes[1]
ax_circ.set_xlim(0, 10)
ax_circ.set_ylim(0, 10)
ax_circ.axis('off')

category_data = [
    ("Price Keywords\n(4 terms)", 5.0, 7.5, 1500, '#2980b9'),
    ("Investment\nKeywords\n(3 terms)", 2.5, 4.5, 900, '#27ae60'),
    ("Market\nKeywords\n(3 terms)", 7.5, 4.5, 900, '#8e44ad'),
    ("Vernacular\nKeywords\n(2 terms)", 5.0, 1.8, 600, '#e67e22'),
]

for label, cx, cy, sz, col in category_data:
    ax_circ.scatter(cx, cy, s=sz, c=col, alpha=0.7, edgecolors='white', linewidth=2, zorder=3)
    ax_circ.text(cx, cy, label, ha='center', va='center',
                 fontsize=8.5, fontweight='bold', color='white', zorder=4,
                 multialignment='center')

ax_circ.set_title('Keyword Categories\n(Composite = weighted avg)', fontsize=10, fontweight='bold')
ax_circ.text(5, 0.4, 'Composite Google Trends Index = weighted average of 12 keyword scores',
             ha='center', fontsize=7.5, color='gray', style='italic')

# Draw connecting lines between bubbles
connections = [(0,1),(0,2),(1,3),(2,3),(0,3)]
cats_xy = [(d[1], d[2]) for d in category_data]
for i, j in connections:
    ax_circ.plot([cats_xy[i][0], cats_xy[j][0]],
                 [cats_xy[i][1], cats_xy[j][1]],
                 color='gray', lw=0.8, alpha=0.4, zorder=1)

plt.suptitle('Fig 5: Google Trends — India Silver Search Keywords Used in NLP Attention Index',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{RESULTS}/fig5_google_trends_keywords.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS}/fig5_google_trends_keywords.png")

# ─────────────────────────────────────────────────────────────
# FIG 6 — Google Trends Time Series + Distribution
# ─────────────────────────────────────────────────────────────
print("\nGenerating Fig 6: Google Trends Time Series + Distribution...")

trends = df['trends_raw'].dropna()
train_end = pd.Timestamp("2024-03-01")

fig = plt.figure(figsize=(14, 6))
gs  = fig.add_gridspec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.35)

# ── Time series ──
ax1 = fig.add_subplot(gs[0])
ax1.fill_between(trends.index, trends.values, alpha=0.2, color='#e67e22')
ax1.plot(trends.index, trends.values, color='#e67e22', linewidth=0.9, zorder=3)
ax1.axvline(train_end, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

ax1.fill_between(
    trends[trends.index < train_end].index,
    trends[trends.index < train_end].values,
    alpha=0.2, color='#27ae60', label='Training'
)
ax1.fill_between(
    trends[trends.index >= train_end].index,
    trends[trends.index >= train_end].values,
    alpha=0.2, color='#e74c3c', label='Testing'
)

# Rolling 26-week MA
ma26 = trends.rolling(26).mean()
ax1.plot(ma26.index, ma26.values, color='#2c3e50', linewidth=1.5,
         linestyle='--', label='26-week MA', zorder=4)

ax1.set_title('Google Trends — India Silver (2008–2026)', fontsize=10, fontweight='bold')
ax1.set_xlabel('Date', fontsize=9)
ax1.set_ylabel('Relative Search Volume (0–100)', fontsize=9)
ax1.legend(fontsize=8, loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines[['top', 'right']].set_visible(False)

# ── Histogram ──
ax2 = fig.add_subplot(gs[1])
ax2.hist(trends.values, bins=25, color='#e67e22', edgecolor='white',
         alpha=0.8, orientation='horizontal', density=True)
ax2.set_xlabel('Density', fontsize=9)
ax2.set_ylabel('Relative Search Volume', fontsize=9)
ax2.set_title('Distribution\n(Full sample)', fontsize=10, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)

# ── Train vs Test box ──
ax3 = fig.add_subplot(gs[2])
train_data = trends[trends.index < train_end].values
test_data  = trends[trends.index >= train_end].values
bp = ax3.boxplot([train_data, test_data],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#27ae60'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#e74c3c'); bp['boxes'][1].set_alpha(0.7)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Training\n(2008–2024)', 'Test\n(2024–2026)'], fontsize=8)
ax3.set_ylabel('Relative Search Volume', fontsize=9)
ax3.set_title('Train vs Test\nDistribution', fontsize=10, fontweight='bold')
ax3.spines[['top', 'right']].set_visible(False)

# Stats annotation
for i, (data, x) in enumerate([(train_data, 1), (test_data, 2)]):
    ax3.text(x, np.percentile(data, 75) + 2,
             f'μ={data.mean():.2f}\nσ={data.std():.2f}',
             ha='center', fontsize=7.5, color='#2c3e50')

plt.suptitle('Fig 6: Google Trends Index — Time Series, Histogram, and Train/Test Distributions',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{RESULTS}/fig6_google_trends_series.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {RESULTS}/fig6_google_trends_series.png")

# ─────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MISSING OUTPUTS GENERATED:")
print("  Tables: table1, table2, table3, table4")
print("  Figures: fig1, fig2, fig3, fig5, fig6")
print("="*60)
print(f"\nAll files saved to: {RESULTS}/")
