"""
BUILD MASTER DATASET — Indian Market (2015–2026, daily / business-day)
=======================================================================
Mirrors build_master.py but at daily (business-day) frequency from 2015.

Key differences vs weekly pipeline:
  - Resampled to 'B' (business days) instead of 'W-SUN'
  - START_DATE = 2015-01-01 (aligned with GDELT coverage)
  - GDELT sentiment is merged directly here from the three pre-downloaded
    CSVs (no separate step0 needed for the daily pipeline):
            bquxjob_178470f6_19d57f8b1bb.csv  (2015-02 → 2018-12)
            bquxjob_12d5030e_19d57fb8d7d.csv  (2019-01 → 2022-12)
            bquxjob_49489022_19d57fc378f.csv  (2023-01 → 2026-03)
    Each file has columns: date, metal, avg_tone, avg_positive,
    avg_negative, avg_net_sentiment, article_count.
    Gold and silver are kept as separate sentiment columns.
  - Google Trends (weekly) forward-filled to daily
  - EPU (monthly) forward-filled to daily

Output:
    silver/data/master_daily_prices.csv
  columns: date, mcx_silver, gold_usd, brent, usdinr, nifty50,
           vix_india, mcx_gold, geo_risk, trends_raw, india_epu,
           sentiment_silver

HOW TO RUN (from the silver/ directory):
    cd silver
  python build_master_daily.py

Bloomberg k/M suffix convention: k=×1,000 ; M=×1,000,000
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths (run from the silver/code/ directory) ───────────────────────────────
BLOOMBERG_FILE    = "../../common_data/RC DATA.xlsx"
RAW_DATA_DIR      = "../../common_data"
PROCESSED_DATA_DIR = "../data"
YFINANCE_FILES = {
    "gold_usd": f"{RAW_DATA_DIR}/gold.csv",
    "brent":    f"{RAW_DATA_DIR}/brent_crude.csv",
}
TRENDS_FILE = "../../common_data/trends_india.csv"
EPU_FILE    = f"{RAW_DATA_DIR}/India_Policy_Uncertainty_Data.xlsx"
GDELT_FILES = [
    "../../common_data/bquxjob_178470f6_19d57f8b1bb.csv",
    "../../common_data/bquxjob_12d5030e_19d57fb8d7d.csv",
    "../../common_data/bquxjob_49489022_19d57fc378f.csv",
]
OUT_FILE   = f"{PROCESSED_DATA_DIR}/master_daily_prices.csv"
START_DATE = "2015-01-01"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("=" * 60)
print("BUILD MASTER DATASET (Indian market, 2015–2026, daily/B)")
print("=" * 60)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_yfinance_daily(path, col_name):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    series = df.iloc[:, 0].rename(col_name)
    return pd.to_numeric(series, errors="coerce").dropna().resample("B").last()


def parse_bbg(val):
    """Handle Bloomberg k/M suffix."""
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


# ── 1. Bloomberg Excel ────────────────────────────────────────────────────────
print("\n[Bloomberg]")
try:
    bbg = pd.read_excel(BLOOMBERG_FILE, sheet_name="Sheet1")
except FileNotFoundError:
    sys.exit(f"ERROR: {BLOOMBERG_FILE} not found in the silver folder.")

bbg = bbg.rename(columns={"Unnamed: 1": "date"})
bbg["date"] = pd.to_datetime(bbg["date"], errors="coerce")
bbg = bbg.dropna(subset=["date"]).sort_values("date").set_index("date")

for col in bbg.columns:
    if bbg[col].dtype == object and col != "Unnamed: 0":
        bbg[col] = bbg[col].apply(parse_bbg)

bbg_map = {
    "MCXSILV Comdty":    "mcx_silver",
    "MCXGOLD Comdty":    "mcx_gold",
    "GPRXGPRD Index":    "geo_risk",
    "USDINR REGN Curncy":"usdinr",
    "NIFTY Index":       "nifty50",
    "INVIXN Index":      "vix_india",
}

bbg_frames = {}
for bbg_col, col_name in bbg_map.items():
    s = pd.to_numeric(bbg[bbg_col], errors="coerce").dropna().rename(col_name)
    s = s.resample("B").last()
    bbg_frames[col_name] = s
    print(f"  {col_name:<12} {len(s):>5} daily rows  "
          f"{s.index[0].date()} → {s.index[-1].date()}  "
          f"range: {s.min():.2f}–{s.max():.2f}")


# ── 2. yfinance CSVs ─────────────────────────────────────────────────────────
print("\n[yfinance]")
yf_frames = {}
for col_name, path in YFINANCE_FILES.items():
    try:
        s = load_yfinance_daily(path, col_name)
        yf_frames[col_name] = s
        print(f"  {col_name:<10} {len(s):>5} daily rows  "
              f"{s.index[0].date()} → {s.index[-1].date()}  "
              f"range: {s.min():.2f}–{s.max():.2f}")
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} not found in daily/financial_data/pre_processed.")


# ── 3. Google Trends (weekly → daily ffill) ───────────────────────────────────
print("\n[Google Trends]")
try:
    trends_df = pd.read_csv(TRENDS_FILE, parse_dates=["date"], index_col="date")
    trends_wk = trends_df["trends_raw"].rename("trends_raw")
    # Resample weekly → daily business days via forward fill
    trends_daily = trends_wk.resample("B").ffill()
    print(f"  trends_raw  {len(trends_daily):>5} daily rows  "
          f"{trends_daily.index[0].date()} → {trends_daily.index[-1].date()}")
except FileNotFoundError:
    sys.exit("ERROR: trends_india.csv not found in the silver folder.")


# ── 4. India EPU (monthly → daily ffill) ─────────────────────────────────────
print("\n[India EPU]")
epu_raw = pd.read_excel(EPU_FILE)
epu_raw = epu_raw[pd.to_numeric(epu_raw["Year"], errors="coerce").notna()].copy()
epu_raw["Year"]  = epu_raw["Year"].astype(int)
epu_raw["Month"] = epu_raw["Month"].astype(int)
epu_raw["date"]  = pd.to_datetime(epu_raw[["Year", "Month"]].assign(day=1))
epu_raw = epu_raw.set_index("date")["India News-Based Policy Uncertainty Index"].rename("india_epu")
epu_daily = epu_raw.resample("B").ffill()
print(f"  india_epu  {len(epu_daily):>5} daily rows  "
      f"{epu_daily.index[0].date()} → {epu_daily.index[-1].date()}  "
      f"range: {epu_daily.min():.2f}–{epu_daily.max():.2f}")


# ── 5. GDELT Sentiment (three pre-downloaded CSVs) ───────────────────────────
print("\n[GDELT Sentiment]")

gdelt_parts = []
for fpath in GDELT_FILES:
    if not os.path.exists(fpath):
        sys.exit(f"ERROR: GDELT file not found: {fpath}\n"
                 "All three GDELT CSVs must be present in the silver folder.")
    part = pd.read_csv(fpath, parse_dates=["date"])
    gdelt_parts.append(part)

gdelt = pd.concat(gdelt_parts, ignore_index=True)
gdelt["date"] = pd.to_datetime(gdelt["date"])
gdelt = gdelt.sort_values("date").drop_duplicates(subset=["date", "metal"])

print(f"  GDELT total: {len(gdelt)} rows, "
      f"{gdelt['date'].min().date()} → {gdelt['date'].max().date()}")
print(f"  Metals: {sorted(gdelt['metal'].unique())}")

# Normalise avg_tone to [-1, 1] (typical range is -15 to +15)
gdelt["tone_norm"] = gdelt["avg_tone"].clip(-15, 15) / 15.0

# Pivot to wide: one column per metal
def _make_sentiment(metal_name, out_col):
    sub = gdelt[gdelt["metal"] == metal_name].copy()
    sub = sub.set_index("date")[["tone_norm", "article_count"]].rename(
        columns={"tone_norm": "tone"})
    # Resample to business days: weight by article_count where available
    sub_B = sub.resample("B").apply(
        lambda g: (
            (g["tone"] * g["article_count"]).sum() / g["article_count"].sum()
            if g["article_count"].sum() > 0 else np.nan
        )
    )
    sub_B.name = out_col
    return sub_B

sent_silver = _make_sentiment("silver", "sentiment_silver")

for s, name in [(sent_silver, "sentiment_silver")]:
    non_na = s.notna().sum()
    print(f"  {name:<18} {len(s):>5} daily rows, {non_na} non-NaN "
          f"({100*non_na/len(s):.1f}%)")


# ── 6. Merge all series ───────────────────────────────────────────────────────
col_order = [
    "mcx_silver", "gold_usd", "brent", "usdinr", "nifty50",
    "vix_india", "mcx_gold", "geo_risk", "trends_raw", "india_epu",
    "sentiment_silver",
]

all_series = {
    **bbg_frames,
    **yf_frames,
    "trends_raw":       trends_daily,
    "india_epu":        epu_daily,
    "sentiment_silver": sent_silver,
}

merged = pd.DataFrame({name: all_series[name] for name in col_order})

# Filter to 2015 onwards
merged = merged[merged.index >= START_DATE]

# Forward-fill gaps (holidays, non-trading days) — limit 3 days for sentiment,
# 5 days for prices (max over a calendar week)
price_cols = ["mcx_silver", "gold_usd", "brent", "usdinr", "nifty50",
              "vix_india", "mcx_gold", "geo_risk"]
slow_cols  = ["trends_raw", "india_epu"]
sent_cols  = ["sentiment_silver"]

merged[price_cols] = merged[price_cols].ffill(limit=5)
merged[slow_cols]  = merged[slow_cols].ffill(limit=31)   # monthly EPU / weekly trends
merged[sent_cols]  = merged[sent_cols].ffill(limit=3).bfill(limit=5)

# Drop rows still missing the target
merged = merged.dropna(subset=["mcx_silver"])
merged.index.name = "date"

# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"\nMerged dataset: {merged.shape[0]} business-day rows × {merged.shape[1]} columns")
print(f"Date range    : {merged.index[0].date()} → {merged.index[-1].date()}")
print(f"NaN per column:\n{merged.isna().sum().to_string()}")
print("\nDescriptive statistics:")
print(merged.describe().round(2).to_string())

# ── 8. Save ───────────────────────────────────────────────────────────────────
merged.to_csv(OUT_FILE)
print(f"\nSaved: {OUT_FILE}")
print("NEXT: python step1_vmd_decompose_daily.py")
