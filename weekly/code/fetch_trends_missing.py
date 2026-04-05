"""
Fetch the 4 missing Google Trends keywords and merge into trends_india.csv
Missing batch: silver ETF India, precious metals India, silver demand India, silver futures India
"""

import time
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

GEO       = "IN"
TIMEFRAME = "2004-01-01 2026-03-01"
ANCHOR    = "silver price India"

MISSING_BATCH = [
    "silver price India",      # anchor
    "silver ETF India",
    "precious metals India",
    "silver demand India",
    "silver futures India",
]

print("Fetching missing batch:", MISSING_BATCH[1:])

pytrends = TrendReq(hl="en-US", tz=330)

# Retry with longer back-off
for attempt in range(5):
    try:
        pytrends.build_payload(MISSING_BATCH, cat=0, timeframe=TIMEFRAME, geo=GEO)
        df = pytrends.interest_over_time()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        print(f"Got {len(df)} rows on attempt {attempt+1}")
        break
    except Exception as e:
        wait = 90 * (attempt + 1)
        print(f"  Attempt {attempt+1} failed: {e} — waiting {wait}s...")
        time.sleep(wait)
else:
    raise RuntimeError("All attempts failed. Try again later.")

# Load existing trends_india.csv to get anchor reference
existing = pd.read_csv("../../preprocessed_data/processed/trends_india.csv", parse_dates=["date"], index_col="date")

# Load the raw monthly series already fetched (from the original run)
# Re-fetch anchor batch to get the anchor reference series
print("Fetching anchor batch for normalisation...")
time.sleep(15)
anchor_batch = [
    "silver price India",
    "MCX silver",
    "silver rate today",
    "chandi price",
    "silver investment India",
]
for attempt in range(5):
    try:
        pytrends.build_payload(anchor_batch, cat=0, timeframe=TIMEFRAME, geo=GEO)
        anchor_df = pytrends.interest_over_time()
        if "isPartial" in anchor_df.columns:
            anchor_df = anchor_df.drop(columns=["isPartial"])
        print(f"Anchor batch fetched ({len(anchor_df)} rows)")
        break
    except Exception as e:
        wait = 90 * (attempt + 1)
        print(f"  Attempt {attempt+1} failed: {e} — waiting {wait}s...")
        time.sleep(wait)
else:
    raise RuntimeError("Anchor batch failed. Try again later.")

anchor_series = anchor_df[ANCHOR]

# Normalise missing batch to anchor
local_anchor = df[ANCHOR].replace(0, np.nan)
ref_anchor   = anchor_series.replace(0, np.nan)
aligned = pd.concat([local_anchor, ref_anchor], axis=1).dropna()
scale = aligned.iloc[:, 1].mean() / aligned.iloc[:, 0].mean() if len(aligned) > 0 else 1.0

new_series = {}
for col in df.columns:
    if col != ANCHOR:
        new_series[col] = df[col] * scale
        print(f"  Normalised: {col}")

# Combine all 12 keywords: existing 8 + new 4
# Reconstruct monthly combined from anchor_df + df
all_monthly = pd.concat([anchor_df.drop(columns=[ANCHOR]),
                          pd.DataFrame(new_series)], axis=1)
all_monthly = all_monthly.fillna(0)
combined_monthly = pd.concat([anchor_series, all_monthly], axis=1).mean(axis=1)

print(f"\nCombined {1 + len(anchor_batch) - 1 + len(new_series)} keyword series (all 12)")

# Resample to weekly
START_WEEKLY = "2000-01-02"
END_WEEKLY   = "2026-03-15"
weekly_idx = pd.date_range(start=START_WEEKLY, end=END_WEEKLY, freq="W-SUN")

trends_weekly = (
    combined_monthly
    .reindex(combined_monthly.index.union(weekly_idx))
    .interpolate(method="time")
    .reindex(weekly_idx)
    .fillna(0)
    .clip(lower=0)
)

max_val = trends_weekly.max()
if max_val > 0:
    trends_weekly = trends_weekly / max_val * 100

trends_weekly.index.name = "date"
out_df = trends_weekly.reset_index()
out_df.columns = ["date", "trends_raw"]
out_df.to_csv("../../preprocessed_data/processed/trends_india.csv", index=False)

print(f"\nUpdated: trends_india.csv ({len(out_df)} weekly rows)")
print(f"trends_raw stats:\n{out_df['trends_raw'].describe().round(2)}")
print("\nAll 12 keywords now included:")
all_kw = list(anchor_batch) + list(new_series.keys())
for kw in all_kw:
    print(f"  • {kw}")
