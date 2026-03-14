"""
BUILD MASTER DATASET — Indian Market (25-year weekly)
======================================================
Reads downloaded CSVs and merges into master_weekly_prices.csv

Input files (run download scripts first):
  financial_data/silver.csv
  financial_data/gold.csv
  financial_data/brent_crude.csv
  financial_data/nifty50.csv       (replaces sp500)
  financial_data/usdinr.csv        (replaces dxy)
  vix.csv
  trends_india.csv                 (run fetch_trends.py first)

Output:
  master_weekly_prices.csv
  columns: date, silver, gold, brent, usdinr, nifty50, vix, trends_raw

HOW TO RUN:
  python build_master.py

All prices are resampled to weekly (W-SUN) using last observation.
Rows with any NaN are dropped. Date range is 2000-01 → 2026-03.
"""

import pandas as pd
import numpy as np

PRICE_FILES = {
    "silver":  "financial_data/silver.csv",
    "gold":    "financial_data/gold.csv",
    "brent":   "financial_data/brent_crude.csv",
    "nifty50": "financial_data/nifty50.csv",
    "usdinr":  "financial_data/usdinr.csv",
}
VIX_FILE    = "vix.csv"
TRENDS_FILE = "trends_india.csv"
OUT_FILE    = "master_weekly_prices.csv"

print("=" * 55)
print("BUILD MASTER DATASET (Indian market, 25 years)")
print("=" * 55)


def load_price(path, col_name):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    series = df.iloc[:, 0].rename(col_name)
    series = pd.to_numeric(series, errors="coerce").dropna()
    # Resample to weekly W-SUN using last observation
    return series.resample("W-SUN").last()


# ── Load price series ─────────────────────────────────────────
frames = {}
for name, path in PRICE_FILES.items():
    try:
        frames[name] = load_price(path, name)
        print(f"  Loaded {name}: {len(frames[name])} weekly rows  "
              f"({frames[name].index[0].date()} – {frames[name].index[-1].date()})")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run download_financial_data.py first."
        )

# ── Load VIX ──────────────────────────────────────────────────
try:
    vix_df = pd.read_csv(VIX_FILE, parse_dates=["date"], index_col="date")
    vix = pd.to_numeric(vix_df.iloc[:, 0], errors="coerce").dropna().rename("vix")
    vix_weekly = vix.resample("W-SUN").last()
    print(f"  Loaded vix   : {len(vix_weekly)} weekly rows  "
          f"({vix_weekly.index[0].date()} – {vix_weekly.index[-1].date()})")
except FileNotFoundError:
    raise FileNotFoundError("Missing vix.csv — run fetch_vix.py first.")

# ── Load Trends ───────────────────────────────────────────────
try:
    trends_df = pd.read_csv(TRENDS_FILE, parse_dates=["date"], index_col="date")
    trends = trends_df["trends_raw"].rename("trends_raw")
    print(f"  Loaded trends: {len(trends)} weekly rows  "
          f"({trends.index[0].date()} – {trends.index[-1].date()})")
except FileNotFoundError:
    raise FileNotFoundError(
        "Missing trends_india.csv — run fetch_trends.py first.\n"
        "If pytrends is unavailable, copy your manual Trends CSV here "
        "and rename it trends_india.csv with columns: date, trends_raw"
    )

# ── Merge all on weekly date (inner join) ─────────────────────
merged = pd.DataFrame(index=frames["silver"].index)
for name, series in frames.items():
    merged = merged.join(series, how="inner")
merged = merged.join(vix_weekly, how="inner")
merged = merged.join(trends, how="inner")

# Forward-fill up to 2 weeks for weekends/holidays in VIX / Nifty
merged = merged.ffill(limit=2)
merged = merged.dropna()
merged.index.name = "date"

# ── Column order matches rest of pipeline ─────────────────────
col_order = ["silver", "gold", "brent", "usdinr", "nifty50", "vix", "trends_raw"]
merged = merged[col_order]

# ── Summary ───────────────────────────────────────────────────
print(f"\nMerged dataset: {merged.shape[0]} weekly rows × {merged.shape[1]} columns")
print(f"Date range    : {merged.index[0].date()} → {merged.index[-1].date()}")
print(f"Columns       : {list(merged.columns)}")
print("\nDescriptive statistics:")
print(merged.describe().round(2).to_string())

# ── Save ─────────────────────────────────────────────────────
merged.to_csv(OUT_FILE)
print(f"\nSaved: {OUT_FILE}")
print("NEXT: python step1_vmd_decompose.py")
