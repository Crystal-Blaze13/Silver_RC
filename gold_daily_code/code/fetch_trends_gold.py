"""
fetch_trends_gold.py — Fetch India-focused gold Google Trends data (12 keywords)
=================================================================================
Gold adaptation of fetch_trends.py (silver).

Uses pytrends to download monthly search interest from Google Trends
for 12 India-relevant gold keywords, normalises across batches using
an anchor keyword, then combines into a single weekly index saved as
trends_india_gold.csv.

Keywords cover:
  - Price discovery (general + India + Hindi)
  - Investment / ETF / Sovereign Gold Bond demand
  - Jewellery demand (India's dominant use case)
  - Futures / MCX trading
  - Retail purchase intent

pip install pytrends

HOW TO RUN:
  python fetch_trends_gold.py

OUTPUT:
  trends_india_gold.csv   (columns: date, trends_raw)
  date = Sunday weekly frequency (W-SUN), 2000-2026
  trends_raw = normalised combined index, 0-100
"""

import time
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

# ── Settings ───────────────────────────────────────────────────────────────────
GEO       = "IN"                      # India
TIMEFRAME = "2004-01-01 2026-03-01"   # monthly back to 2004
ANCHOR    = "gold price India"        # cross-normalisation anchor
CAT       = 0                         # all categories

# ── 12 India-relevant gold keywords ───────────────────────────────────────────
KEYWORDS = [
    "gold price India",        # 1 — anchor + primary search
    "MCX gold",                # 2 — MCX futures trading
    "gold rate today",         # 3 — daily retail inquiry
    "sona price",              # 4 — Hindi term (sona = gold)
    "gold investment India",   # 5 — investment demand
    "gold ETF India",          # 6 — ETF demand
    "sovereign gold bond",     # 7 — India-specific SGB scheme
    "gold demand India",       # 8 — industrial/jewellery demand
    "gold futures India",      # 9 — derivatives market
    "gold jewellery India",    # 10 — jewellery sector (India's largest use)
    "buy gold India",          # 11 — retail purchase intent
    "gold rate per gram",      # 12 — Indian retail unit (grams, not oz)
]


def make_batches(keywords, anchor, batch_size=4):
    """
    Split keywords (excluding anchor) into batches of `batch_size`.
    Each batch is prepended with the anchor for cross-normalisation.
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
            time.sleep(60 * (attempt + 1))
    return pd.DataFrame()


def normalise_batch(batch_df, anchor_series):
    """
    Scale each keyword in batch_df so the anchor column aligns with
    anchor_series (the reference batch's anchor values).
    """
    anchor_col   = batch_df.columns[0]
    local_anchor = batch_df[anchor_col].replace(0, np.nan)
    ref_anchor   = anchor_series.replace(0, np.nan)

    aligned = pd.concat([local_anchor, ref_anchor], axis=1).dropna()
    scale   = (aligned.iloc[:, 1].mean() / aligned.iloc[:, 0].mean()
               if len(aligned) > 0 else 1.0)

    result = {}
    for col in batch_df.columns:
        result[col] = batch_df[col] * scale
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 55)
print("Fetching India gold Google Trends (12 keywords)")
print("=" * 55)

pytrends = TrendReq(hl="en-US", tz=330)   # IST offset

batches = make_batches(KEYWORDS, ANCHOR, batch_size=4)
print(f"\nKeywords: {len(KEYWORDS)}")
print(f"Batches : {len(batches)}")

# ── Fetch anchor batch first ───────────────────────────────────────────────────
print(f"\nBatch 0 (anchor): {batches[0]}")
anchor_df = fetch_batch(pytrends, batches[0], TIMEFRAME, GEO)

if anchor_df.empty:
    raise RuntimeError("Failed to fetch anchor batch. Check pytrends / network.")

anchor_series = anchor_df[ANCHOR]
print(f"  Got {len(anchor_df)} monthly rows, "
      f"{anchor_df.index[0].date()} – {anchor_df.index[-1].date()}")

all_series = {}
for col in anchor_df.columns:
    all_series[col] = anchor_df[col]

# ── Fetch remaining batches ────────────────────────────────────────────────────
for b_idx, batch in enumerate(batches[1:], start=1):
    print(f"\nBatch {b_idx}: {batch}")
    time.sleep(10)

    df = fetch_batch(pytrends, batch, TIMEFRAME, GEO)
    if df.empty:
        print(f"  WARNING: batch {b_idx} returned empty — skipping")
        continue

    print(f"  Got {len(df)} rows")
    normed = normalise_batch(df, anchor_series)
    for kw, series in normed.items():
        if kw != ANCHOR:
            all_series[kw] = series

# ── Combine into single index ──────────────────────────────────────────────────
print(f"\nCombining {len(all_series)} keyword series into composite index...")

combined_df      = pd.DataFrame(all_series).fillna(0)
combined_monthly = combined_df.mean(axis=1)
combined_monthly.name = "trends_raw"

print(f"Monthly index: {len(combined_monthly)} rows, "
      f"{combined_monthly.index[0].date()} – {combined_monthly.index[-1].date()}")

# ── Resample monthly → weekly (W-SUN) via linear interpolation ────────────────
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
out_df.to_csv("../../common_data/trends_india_gold.csv", index=False)

print(f"\nSaved: trends_india_gold.csv  ({len(out_df)} weekly rows)")
print(f"Range: {out_df['date'].min()} → {out_df['date'].max()}")
print(f"trends_raw stats:\n{out_df['trends_raw'].describe().round(2)}")
print(f"\nKeywords used ({len(all_series)}):")
for kw in all_series:
    print(f"  • {kw}")
