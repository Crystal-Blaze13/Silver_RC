"""
BUILD MASTER DATASET — Indian Market (2015–2026, daily / business-day) — GOLD
==============================================================================
Gold adaptation of build_master_daily.py (silver).

Key differences vs silver master:
  - TARGET column: mcx_gold  (instead of mcx_silver)
  - Cross-metal feature: silver_usd (COMEX silver, from yfinance silver.csv)
  - Sentiment: sentiment_gold  (GDELT avg_tone filtered for metal="gold")
  - Google Trends: trends_india_gold.csv  (gold keyword index)
  - Output: master_daily_prices_gold.csv

Column order:
  date, mcx_gold, silver_usd, brent, usdinr, nifty50,
  vix_india, mcx_silver, geo_risk, trends_raw, india_epu, sentiment_gold

HOW TO RUN (from the daily/code/ directory):
  python build_master_daily_gold.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BLOOMBERG_FILE     = "../../common_data/RC DATA.xlsx"
RAW_DATA_DIR       = "../../common_data"
PROCESSED_DATA_DIR = "../../common_data"

YFINANCE_FILES = {
    "silver_usd": f"{RAW_DATA_DIR}/silver.csv",   # COMEX silver (cross-metal)
    "brent":      f"{RAW_DATA_DIR}/brent_crude.csv",
}
TRENDS_FILE = f"{RAW_DATA_DIR}/trends_india_gold.csv"   # gold-specific trends
EPU_FILE    = f"{RAW_DATA_DIR}/India_Policy_Uncertainty_Data.xlsx"
GDELT_FILES = [
    f"{RAW_DATA_DIR}/bquxjob_178470f6_19d57f8b1bb.csv",
    f"{RAW_DATA_DIR}/bquxjob_12d5030e_19d57fb8d7d.csv",
    f"{RAW_DATA_DIR}/bquxjob_49489022_19d57fc378f.csv",
]
OUT_FILE   = "../data/master_daily_prices_gold.csv"
START_DATE = "2015-01-01"

os.makedirs("../data", exist_ok=True)

print("=" * 60)
print("BUILD MASTER DATASET — GOLD (Indian market, 2015–2026, daily/B)")
print("=" * 60)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_yfinance_daily(path, col_name):
    df     = pd.read_csv(path, parse_dates=["date"], index_col="date")
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
    sys.exit(f"ERROR: {BLOOMBERG_FILE} not found.")

bbg = bbg.rename(columns={"Unnamed: 1": "date"})
bbg["date"] = pd.to_datetime(bbg["date"], errors="coerce")
bbg = bbg.dropna(subset=["date"]).sort_values("date").set_index("date")

for col in bbg.columns:
    if bbg[col].dtype == object and col != "Unnamed: 0":
        bbg[col] = bbg[col].apply(parse_bbg)

# NOTE: mcx_gold is the TARGET; mcx_silver kept as a cross-metal feature
bbg_map = {
    "MCXGOLD Comdty":    "mcx_gold",      # ← TARGET
    "MCXSILV Comdty":    "mcx_silver",    # cross-metal feature
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


# ── 2. yfinance CSVs (silver_usd cross-metal + brent) ────────────────────────
print("\n[yfinance]")
yf_frames = {}
for col_name, path in YFINANCE_FILES.items():
    try:
        s = load_yfinance_daily(path, col_name)
        yf_frames[col_name] = s
        print(f"  {col_name:<12} {len(s):>5} daily rows  "
              f"{s.index[0].date()} → {s.index[-1].date()}  "
              f"range: {s.min():.2f}–{s.max():.2f}")
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} not found.")


# ── 3. Google Trends gold (weekly → daily ffill) ──────────────────────────────
print("\n[Google Trends — Gold]")
try:
    trends_df    = pd.read_csv(TRENDS_FILE, parse_dates=["date"], index_col="date")
    trends_wk    = trends_df["trends_raw"].rename("trends_raw")
    trends_daily = trends_wk.resample("B").ffill()
    print(f"  trends_raw  {len(trends_daily):>5} daily rows  "
          f"{trends_daily.index[0].date()} → {trends_daily.index[-1].date()}")
except FileNotFoundError:
    sys.exit("ERROR: trends_india_gold.csv not found. Run fetch_trends_gold.py first.")


# ── 4. India EPU (monthly → daily ffill) ─────────────────────────────────────
print("\n[India EPU]")
epu_raw = pd.read_excel(EPU_FILE)
epu_raw = epu_raw[pd.to_numeric(epu_raw["Year"], errors="coerce").notna()].copy()
epu_raw["Year"]  = epu_raw["Year"].astype(int)
epu_raw["Month"] = epu_raw["Month"].astype(int)
epu_raw["date"]  = pd.to_datetime(epu_raw[["Year", "Month"]].assign(day=1))
epu_raw  = epu_raw.set_index("date")["India News-Based Policy Uncertainty Index"].rename("india_epu")
epu_daily = epu_raw.resample("B").ffill()
print(f"  india_epu  {len(epu_daily):>5} daily rows  "
      f"{epu_daily.index[0].date()} → {epu_daily.index[-1].date()}  "
      f"range: {epu_daily.min():.2f}–{epu_daily.max():.2f}")


# ── 5. GDELT Sentiment — GOLD rows ────────────────────────────────────────────
print("\n[GDELT Sentiment — Gold]")

gdelt_parts = []
for fpath in GDELT_FILES:
    if not os.path.exists(fpath):
        sys.exit(f"ERROR: GDELT file not found: {fpath}")
    part = pd.read_csv(fpath, parse_dates=["date"])
    gdelt_parts.append(part)

gdelt = pd.concat(gdelt_parts, ignore_index=True)
gdelt["date"] = pd.to_datetime(gdelt["date"])
gdelt = gdelt.sort_values("date").drop_duplicates(subset=["date", "metal"])

print(f"  GDELT total: {len(gdelt)} rows, "
      f"{gdelt['date'].min().date()} → {gdelt['date'].max().date()}")
print(f"  Metals: {sorted(gdelt['metal'].unique())}")

gdelt["tone_norm"] = gdelt["avg_tone"].clip(-15, 15) / 15.0


def _make_sentiment(metal_name, out_col):
    sub  = gdelt[gdelt["metal"] == metal_name].copy()
    sub  = sub.set_index("date")[["tone_norm", "article_count"]].rename(
        columns={"tone_norm": "tone"})
    sub_B = sub.resample("B").apply(
        lambda g: (
            (g["tone"] * g["article_count"]).sum() / g["article_count"].sum()
            if g["article_count"].sum() > 0 else np.nan
        )
    )
    sub_B.name = out_col
    return sub_B


sent_gold = _make_sentiment("gold", "sentiment_gold")

non_na = sent_gold.notna().sum()
print(f"  sentiment_gold     {len(sent_gold):>5} daily rows, {non_na} non-NaN "
      f"({100*non_na/len(sent_gold):.1f}%)")


# ── 6. Merge all series ───────────────────────────────────────────────────────
col_order = [
    "mcx_gold", "silver_usd", "brent", "usdinr", "nifty50",
    "vix_india", "mcx_silver", "geo_risk", "trends_raw", "india_epu",
    "sentiment_gold",
]

all_series = {
    **bbg_frames,
    **yf_frames,
    "trends_raw":    trends_daily,
    "india_epu":     epu_daily,
    "sentiment_gold": sent_gold,
}

merged = pd.DataFrame({name: all_series[name] for name in col_order})
merged = merged[merged.index >= START_DATE]

# Forward-fill gaps
price_cols = ["mcx_gold", "silver_usd", "brent", "usdinr", "nifty50",
              "vix_india", "mcx_silver", "geo_risk"]
slow_cols  = ["trends_raw", "india_epu"]
sent_cols  = ["sentiment_gold"]

merged[price_cols] = merged[price_cols].ffill(limit=5)
merged[slow_cols]  = merged[slow_cols].ffill(limit=31)
merged[sent_cols]  = merged[sent_cols].ffill(limit=3).bfill(limit=5)

merged = merged.dropna(subset=["mcx_gold"])
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
print("NEXT: python step1_vmd_decompose_daily_gold.py")
