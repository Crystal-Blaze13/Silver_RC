"""
Fetch India-focused silver Google Trends data (12 keywords)
============================================================
Uses pytrends to download monthly search interest from Google Trends
for 12 India-relevant silver keywords, normalises across batches using
an anchor keyword, then combines into a single weekly index saved as
trends_india.csv.

Monthly resolution is used because Google Trends only provides weekly
data for the most recent 5 years; monthly covers back to 2004-01.
For pre-2004 (2000-2003) no trends data exists — those weeks are
filled with 0 (unknown / no internet penetration in India for silver
search).

Keywords chosen to cover:
  - Price discovery (general + India + Hindi)
  - Investment / ETF demand
  - Jewellery / industrial demand
  - Futures / MCX trading

pip install pytrends

HOW TO RUN:
  python fetch_trends.py

OUTPUT:
  trends_india.csv   (columns: date, trends_raw)
  date = Sunday weekly frequency (W-SUN), 2000-2026
  trends_raw = normalised combined index, 0-100
"""

import time
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

# ── Settings ───────────────────────────────────────────────────
GEO       = "IN"            # India
TIMEFRAME = "2004-01-01 2026-03-01"   # monthly back to 2004
ANCHOR    = "silver price India"      # used to cross-normalise batches
CAT       = 0               # all categories

# ── 12 India-relevant silver keywords (anchor must be first in each batch) ──
KEYWORDS = [
    "silver price India",     # 1 — anchor + primary search
    "MCX silver",             # 2 — MCX futures trading
    "silver rate today",      # 3 — daily retail inquiry
    "chandi price",           # 4 — Hindi term (chandi = silver)
    "silver investment India", # 5 — investment demand
    "silver ETF India",        # 6 — ETF demand
    "precious metals India",   # 7 — broader precious metals
    "silver demand India",     # 8 — industrial/jewellery demand
    "silver futures India",    # 9 — derivatives market
    "silver jewellery India",  # 10 — jewellery sector
    "buy silver India",        # 11 — retail purchase intent
    "silver rate per kg",      # 12 — Indian retail unit (kg, not oz)
]

# Batch into groups of 5, always prepend anchor so we can normalise
def make_batches(keywords, anchor, batch_size=4):
    """
    Split keywords (excluding anchor) into batches of `batch_size`.
    Each batch is prepended with the anchor for cross-normalisation.
    The anchor batch is returned first (all 5 slots).
    """
    non_anchor = [k for k in keywords if k != anchor]
    batches = []
    for i in range(0, len(non_anchor), batch_size):
        chunk = non_anchor[i:i + batch_size]
        batches.append([anchor] + chunk)
    return batches


def fetch_batch(pytrends_obj, kw_list, timeframe, geo, retries=3):
    for attempt in range(retries):
        try:
            pytrends_obj.build_payload(kw_list, cat=CAT,
                                       timeframe=timeframe, geo=geo)
            df = pytrends_obj.interest_over_time()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            return df
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(60 * (attempt + 1))   # back-off
    return pd.DataFrame()


def normalise_batch(batch_df, anchor_series):
    """
    Scale each keyword in batch_df so the anchor column aligns with
    anchor_series (the reference batch's anchor values).
    Returns dict: keyword → normalised Series.
    """
    anchor_col = batch_df.columns[0]
    local_anchor = batch_df[anchor_col].replace(0, np.nan)
    ref_anchor   = anchor_series.replace(0, np.nan)

    # Align on common dates
    aligned = pd.concat([local_anchor, ref_anchor], axis=1).dropna()
    if len(aligned) == 0:
        scale = 1.0
    else:
        scale = aligned.iloc[:, 1].mean() / aligned.iloc[:, 0].mean()

    result = {}
    for col in batch_df.columns:
        result[col] = batch_df[col] * scale
    return result


# ── Main ───────────────────────────────────────────────────────
print("=" * 55)
print("Fetching India silver Google Trends (12 keywords)")
print("=" * 55)

pytrends = TrendReq(hl="en-US", tz=330)   # IST offset

batches = make_batches(KEYWORDS, ANCHOR, batch_size=4)
print(f"\nKeywords: {len(KEYWORDS)}")
print(f"Batches : {len(batches)}")

# ── Fetch anchor batch first ───────────────────────────────────
print(f"\nBatch 0 (anchor): {batches[0]}")
anchor_df = fetch_batch(pytrends, batches[0], TIMEFRAME, GEO)

if anchor_df.empty:
    raise RuntimeError("Failed to fetch anchor batch. Check pytrends / network.")

anchor_series = anchor_df[ANCHOR]
print(f"  Got {len(anchor_df)} monthly rows, {anchor_df.index[0].date()} – {anchor_df.index[-1].date()}")

all_series = {}
for col in anchor_df.columns:
    all_series[col] = anchor_df[col]

# ── Fetch remaining batches ───────────────────────────────────
for b_idx, batch in enumerate(batches[1:], start=1):
    print(f"\nBatch {b_idx}: {batch}")
    time.sleep(10)   # polite delay between requests

    df = fetch_batch(pytrends, batch, TIMEFRAME, GEO)
    if df.empty:
        print(f"  WARNING: batch {b_idx} returned empty — skipping")
        continue

    print(f"  Got {len(df)} rows")
    normed = normalise_batch(df, anchor_series)
    for kw, series in normed.items():
        if kw != ANCHOR:   # don't add anchor twice
            all_series[kw] = series

# ── Combine into single index ─────────────────────────────────
print(f"\nCombining {len(all_series)} keyword series into composite index...")

combined_df = pd.DataFrame(all_series)
# Fill missing with 0 (pre-2004 will be absent anyway)
combined_df = combined_df.fillna(0)
combined_monthly = combined_df.mean(axis=1)
combined_monthly.name = "trends_raw"

print(f"Monthly index: {len(combined_monthly)} rows, "
      f"{combined_monthly.index[0].date()} – {combined_monthly.index[-1].date()}")

# ── Resample monthly → weekly (W-SUN) via linear interpolation ──
START_WEEKLY = "2000-01-02"   # first Sunday on/after 2000-01-01
END_WEEKLY   = "2026-03-15"

weekly_idx = pd.date_range(start=START_WEEKLY, end=END_WEEKLY, freq="W-SUN")

# Reindex to weekly, interpolate, then fill pre-2004 with 0
trends_weekly = (
    combined_monthly
    .reindex(combined_monthly.index.union(weekly_idx))
    .interpolate(method="time")
    .reindex(weekly_idx)
    .fillna(0)
    .clip(lower=0)
)

# Rescale to 0-100
max_val = trends_weekly.max()
if max_val > 0:
    trends_weekly = trends_weekly / max_val * 100

trends_weekly.index.name = "date"
out_df = trends_weekly.reset_index()
out_df.columns = ["date", "trends_raw"]
out_df.to_csv("../../preprocessed_data/processed/trends_india.csv", index=False)

print(f"\nSaved: trends_india.csv  ({len(out_df)} weekly rows)")
print(f"Range: {out_df['date'].min()} → {out_df['date'].max()}")
print(f"trends_raw stats:\n{out_df['trends_raw'].describe().round(2)}")
print(f"\nKeywords used ({len(all_series)}):")
for kw in all_series:
    print(f"  • {kw}")
