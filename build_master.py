"""
BUILD MASTER DATASET — Indian Market (2008-2026, weekly)
=========================================================
Target variable:
  mcx_silver  — MCX Silver price (MCXSILV Comdty, INR/kg, Bloomberg)
                This is the actual benchmark for India's silver supply chain.

Feature candidates (14):
  Lags 1-5 of mcx_silver
  gold_usd     — CME Gold futures (yfinance GC=F, USD/oz)
  brent        — ICE Brent crude  (yfinance BZ=F, USD/bbl)
  usdinr       — USD/INR spot     (Bloomberg USDINR REGN Curncy)
  nifty50      — Nifty 50 index   (Bloomberg NIFTY Index)
  vix_india    — India VIX        (Bloomberg INVIXN Index)
  mcx_gold     — MCX Gold         (Bloomberg MCXGOLD Comdty, INR/10g; k/M parsed)
  geo_risk     — Geopolitical risk(Bloomberg GPRXGPRD Index)
  trends_raw   — Google Trends    (12 India silver keywords, 0-100 normalised)
  india_epu    — India Economic Policy Uncertainty Index (Baker, Bloom & Davis)

Data sources:
  Bloomberg Excel (RC DATA.xlsx, Sheet1) — k/M suffixes handled by parse_bbg()
  yfinance CSVs  — gold_usd, brent (clean USD, no unit issues)
  Google Trends  — trends_india.csv
    EPU data       — financial_data/pre_processed/India_Policy_Uncertainty_Data.xlsx (monthly → weekly ffill)

Output:
  master_weekly_prices.csv
  columns: date, mcx_silver, gold_usd, brent, usdinr, nifty50,
           vix_india, mcx_gold, geo_risk, trends_raw, india_epu

HOW TO RUN:
  python build_master.py

All prices resampled to weekly (W-SUN), last observation.
Date range: 2008-01-01 onwards (India VIX launched Nov 2007).
Rows with any NaN dropped after forward-fill (limit=2).

Bloomberg k/M suffix convention: k = ×1,000 ; M = ×1,000,000
"""

import pandas as pd
import numpy as np

BLOOMBERG_FILE = "RC DATA.xlsx"
RAW_DATA_DIR = "financial_data/pre_processed"
PROCESSED_DATA_DIR = "financial_data/processed"
YFINANCE_FILES = {
    "gold_usd": f"{RAW_DATA_DIR}/gold.csv",
    "brent":    f"{RAW_DATA_DIR}/brent_crude.csv",
}
TRENDS_FILE = "trends_india.csv"
EPU_FILE    = f"{RAW_DATA_DIR}/India_Policy_Uncertainty_Data.xlsx"
OUT_FILE    = f"{PROCESSED_DATA_DIR}/master_weekly_prices.csv"
START_DATE  = "2008-01-01"

print("=" * 60)
print("BUILD MASTER DATASET (Indian market, 2008–2026, weekly)")
print("=" * 60)


def load_yfinance(path, col_name):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    series = df.iloc[:, 0].rename(col_name)
    return pd.to_numeric(series, errors="coerce").dropna().resample("W-SUN").last()


def parse_bbg(val):
    """Parse Bloomberg numeric values that may carry k (×1000) or M (×1,000,000) suffixes."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper().replace(",", "")
    if s.endswith("K"):
        return float(s[:-1]) * 1_000
    if s.endswith("M"):
        return float(s[:-1]) * 1_000_000
    try:
        return float(s)
    except ValueError:
        return np.nan


# ── Load Bloomberg Excel ───────────────────────────────────────
print("\n[Bloomberg]")
try:
    bbg = pd.read_excel(BLOOMBERG_FILE, sheet_name="Sheet1")
except FileNotFoundError:
    raise FileNotFoundError(f"Missing {BLOOMBERG_FILE} — place Bloomberg Excel in repo root.")

bbg = bbg.rename(columns={"Unnamed: 1": "date"})
bbg["date"] = pd.to_datetime(bbg["date"], errors="coerce")
bbg = bbg.dropna(subset=["date"]).sort_values("date").set_index("date")

# Parse all object-dtype columns to handle Bloomberg k/M suffixes
for col in bbg.columns:
    if bbg[col].dtype == object and col != "Unnamed: 0":
        bbg[col] = bbg[col].apply(parse_bbg)

bbg_map = {
    "MCXSILV Comdty":    "mcx_silver",   # target — MCX Silver INR/kg
    "MCXGOLD Comdty":    "mcx_gold",     # MCX Gold INR/10g (festival/wedding demand driver)
    "GPRXGPRD Index":    "geo_risk",     # Geopolitical risk (India is net silver importer)
    "USDINR REGN Curncy":"usdinr",       # USD/INR — critical for Indian commodity pricing
    "NIFTY Index":       "nifty50",      # Indian equity benchmark
    "INVIXN Index":      "vix_india",    # India VIX (domestic fear gauge)
}

bbg_frames = {}
for bbg_col, col_name in bbg_map.items():
    s = pd.to_numeric(bbg[bbg_col], errors="coerce").dropna().rename(col_name)
    s = s.resample("W-SUN").last()
    bbg_frames[col_name] = s
    print(f"  {col_name:<12} {len(s):>4} weekly rows  "
          f"{s.index[0].date()} → {s.index[-1].date()}  "
          f"range: {s.min():.2f}–{s.max():.2f}")

# ── Load yfinance CSVs ────────────────────────────────────────
print("\n[yfinance]")
yf_frames = {}
for col_name, path in YFINANCE_FILES.items():
    try:
        s = load_yfinance(path, col_name)
        yf_frames[col_name] = s
        print(f"  {col_name:<10} {len(s):>4} weekly rows  "
              f"{s.index[0].date()} → {s.index[-1].date()}  "
              f"range: {s.min():.2f}–{s.max():.2f}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {path} — run download_financial_data.py first.")

# ── Load Trends ───────────────────────────────────────────────
print("\n[Google Trends]")
try:
    trends_df = pd.read_csv(TRENDS_FILE, parse_dates=["date"], index_col="date")
    trends = trends_df["trends_raw"].rename("trends_raw")
    print(f"  trends_raw {len(trends):>4} weekly rows  "
          f"{trends.index[0].date()} → {trends.index[-1].date()}")
except FileNotFoundError:
    raise FileNotFoundError(
        "Missing trends_india.csv — run fetch_trends.py first."
    )

# ── Load India EPU ────────────────────────────────────────────
print("\n[India Economic Policy Uncertainty]")
epu_raw = pd.read_excel(EPU_FILE)
# Drop footer rows (source attribution text in Year column)
epu_raw = epu_raw[pd.to_numeric(epu_raw['Year'], errors='coerce').notna()].copy()
epu_raw['Year']  = epu_raw['Year'].astype(int)
epu_raw['Month'] = epu_raw['Month'].astype(int)
epu_raw['date']  = pd.to_datetime(epu_raw[['Year', 'Month']].assign(day=1))
epu_raw = epu_raw.set_index('date')['India News-Based Policy Uncertainty Index'].rename('india_epu')
# Resample monthly → weekly by forward-fill (each week inherits that month's reading)
epu_weekly = epu_raw.resample('W-SUN').ffill()
print(f"  india_epu  {len(epu_weekly):>4} weekly rows  "
      f"{epu_weekly.index[0].date()} → {epu_weekly.index[-1].date()}  "
      f"range: {epu_weekly.min():.2f}–{epu_weekly.max():.2f}")

# ── Merge all ─────────────────────────────────────────────────
col_order = ["mcx_silver", "gold_usd", "brent", "usdinr", "nifty50",
             "vix_india", "mcx_gold", "geo_risk", "trends_raw", "india_epu"]
all_series = {**bbg_frames, **yf_frames, "trends_raw": trends, "india_epu": epu_weekly}

merged = pd.DataFrame({name: all_series[name] for name in col_order})

# Filter to 2008 onwards
merged = merged[merged.index >= START_DATE]

# Forward-fill up to 2 weeks for holidays / non-trading days
merged = merged.ffill(limit=2)
merged = merged.dropna()
merged.index.name = "date"

# ── Summary ───────────────────────────────────────────────────
print(f"\nMerged dataset: {merged.shape[0]} weekly rows × {merged.shape[1]} columns")
print(f"Date range    : {merged.index[0].date()} → {merged.index[-1].date()}")
print(f"Columns       : {list(merged.columns)}")
print("\nDescriptive statistics:")
print(merged.describe().round(2).to_string())

# ── Save ──────────────────────────────────────────────────────
merged.to_csv(OUT_FILE)
print(f"\nSaved: {OUT_FILE}")
print("NEXT: python step1_vmd_decompose.py")
