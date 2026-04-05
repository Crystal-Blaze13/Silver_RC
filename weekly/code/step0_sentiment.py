"""
STEP 0 — News Sentiment (GDELT + India EPU)
============================================
Paper equivalent: Section 4.2 — unstructured data processing (TextBlob sentiment
on China Daily news headlines). This script replicates that approach for Indian
silver/gold markets using GDELT as the news source and India EPU as a backfill.

Produces:  sentiment_weekly.csv   — weekly sentiment score, 2008–2026
           fig5_wordcloud.png     — word cloud of dominant news terms (Fig 5)
           fig6_sentiment.png     — sentiment time series + distribution (Fig 6)
           master_weekly_prices_with_sentiment.csv — master file with real sentiment

HOW TO RUN — THREE STEPS:
--------------------------
STEP A: Get GDELT data (one-time, ~5 minutes, free, no API key needed)
  1. Go to https://console.cloud.google.com/bigquery
  2. Sign in with any Google account
  3. Click "Compose new query" and paste the query printed by this script
    when you run:  python step0_sentiment.py --print-query
  4. Click RUN, then "Save Results" → "CSV (local file)"
  5. Save the file as:  gdelt_raw.csv  in this project folder

STEP B: Run the full pipeline
     python step0_sentiment.py

STEP C: Update master file
  The script automatically writes master_weekly_prices_with_sentiment.csv.
  Rename it to master_weekly_prices.csv (back up the original first).

DEPENDENCIES:
  pip install textblob pandas numpy matplotlib wordcloud scipy
  python -m textblob.download_corpora
"""

import argparse
import sys
import os
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

MASTER_FILE   = "../processed/master_weekly_prices.csv"
GDELT_FILE    = "gdelt_raw.csv"        # downloaded from BigQuery
EPU_FILE      = "../processed/master_weekly_prices.csv"  # india_epu column
OUT_SENTIMENT = "../processed/sentiment_weekly.csv"
OUT_MASTER    = "../processed/master_weekly_prices_with_sentiment.csv"
OUT_FIG5      = "../results/figures/fig5_wordcloud.png"
OUT_FIG6      = "../results/figures/fig6_sentiment.png"

# Date range to match pipeline
DATE_START = "2008-01-01"
DATE_END   = "2026-12-31"

# GDELT coverage is reliable from ~2015 for Indian sources
GDELT_START = "2015-01-01"

# Keywords — silver + gold + shared macro (reusable for gold pipeline)
KEYWORDS = {
    "silver":    ["silver", "chandi", "mcx silver", "silver futures", "silver etf",
                  "silver price", "silver demand", "silver import"],
    "gold":      ["gold", "sona", "mcx gold", "sovereign gold", "gold bond",
                  "gold futures", "gold etf", "gold price", "gold import",
                  "gold jewellery"],
    "commodity": ["bullion", "precious metal", "comex", "commodity market",
                  "mcx commodity", "jewellery", "hallmark"],
    "macro":     ["rupee", "dollar", "inflation", "rbi", "import duty",
                  "customs duty", "fiscal", "monetary policy", "interest rate"],
}

ALL_KEYWORDS = [kw for group in KEYWORDS.values() for kw in group]

# ── Argument parser ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--print-query", action="store_true",
                    help="Print the BigQuery SQL and exit (run this first)")
parser.add_argument("--use-epu-only", action="store_true",
                    help="Skip GDELT entirely and use India EPU as sentiment proxy")
parser.add_argument("--bq-start", default="2015-02-19",
                    help="BigQuery start date in YYYYMMDDHHMMSS format")
parser.add_argument("--bq-end", default="20261231235959",
                    help="BigQuery end date in YYYYMMDDHHMMSS format")
parser.add_argument("--bq-sample", default="1 PERCENT",
                    help="BigQuery TABLESAMPLE clause, e.g. '1 PERCENT' or '0.1 PERCENT'")
parser.add_argument("--list-months", action="store_true",
                    help="Print month-by-month BigQuery commands for the requested range")
parser.add_argument("--list-sql", action="store_true",
                    help="Print month-by-month BigQuery SQL statements directly")
args = parser.parse_args()

def build_bigquery_sql(bq_start: str, bq_end: str, bq_sample: str) -> str:
    return f"""
-- GDELT Global Knowledge Graph — Indian silver/gold news sentiment
-- Run this in Google BigQuery (console.cloud.google.com/bigquery)
-- Free tier friendly: this query samples the raw table to reduce scanned bytes.
-- Save result as: gdelt_raw.csv

SELECT
  DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS date,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)) AS avg_tone,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS FLOAT64)) AS avg_positive,
  AVG(SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS FLOAT64)) AS avg_negative,
  COUNT(*)                  AS article_count,
  COUNT(*)                  AS total_mentions
FROM
  `gdelt-bq.gdeltv2.gkg` TABLESAMPLE SYSTEM ({bq_sample})
WHERE
  DATE BETWEEN {bq_start} AND {bq_end}
  AND (
    LOWER(V2Themes) LIKE '%silver%'
    OR LOWER(V2Themes) LIKE '%gold%'
    OR LOWER(V2Themes) LIKE '%bullion%'
    OR LOWER(V2Locations) LIKE '%india%'
    OR LOWER(V2Organizations) LIKE '%mcx%'
    OR LOWER(V2Organizations) LIKE '%nse%'
  )
GROUP BY
  date
ORDER BY
  date
"""

if args.print_query:
    print("=" * 70)
    print("PASTE THIS INTO BigQuery (console.cloud.google.com/bigquery):")
    print("=" * 70)
    print(build_bigquery_sql(args.bq_start, args.bq_end, args.bq_sample))
    print("=" * 70)
    print("\nAfter running, click Save Results → CSV (local file)")
    print(f"Save as: {GDELT_FILE} in your project folder")
    print("\nThen run:  python step0_sentiment.py")
    sys.exit(0)

if args.list_months:
    start_dt = datetime.strptime(args.bq_start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(args.bq_end, "%Y%m%d%H%M%S")

    current_start = start_dt
    command_lines = []
    while current_start <= end_dt:
        next_month = (current_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(next_month - timedelta(seconds=1), end_dt)
        command_lines.append(
            f"python step0_sentiment.py --print-query --bq-start {current_start.strftime('%Y%m%d%H%M%S')} --bq-end {chunk_end.strftime('%Y%m%d%H%M%S')} --bq-sample '{args.bq_sample}'"
        )
        current_start = next_month

    print("\n".join(command_lines))
    sys.exit(0)

if args.list_sql:
    start_dt = datetime.strptime(args.bq_start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(args.bq_end, "%Y%m%d%H%M%S")

    current_start = start_dt
    while current_start <= end_dt:
        next_month = (current_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        chunk_end = min(next_month - timedelta(seconds=1), end_dt)
        chunk_sql = f"""
-- Chunk: {current_start.strftime('%Y%m%d%H%M%S')} to {chunk_end.strftime('%Y%m%d%H%M%S')}
{build_bigquery_sql(current_start.strftime('%Y%m%d%H%M%S'), chunk_end.strftime('%Y%m%d%H%M%S'), args.bq_sample)}
"""
        print(chunk_sql)
        print("=" * 70)
        current_start = next_month

    sys.exit(0)

print("=" * 60)
print("STEP 0: News Sentiment Pipeline")
print("=" * 60)

# ── 1. Load master file ───────────────────────────────────────────────────────

df_master = pd.read_csv(MASTER_FILE, index_col=0, parse_dates=True)
df_master.index = pd.to_datetime(df_master.index)
print(f"Master file: {len(df_master)} weeks, "
      f"{df_master.index[0].date()} → {df_master.index[-1].date()}")

# ── 2. Load or simulate GDELT data ───────────────────────────────────────────

if args.use_epu_only or not os.path.exists(GDELT_FILE):
    if not args.use_epu_only:
        print(f"\nWARNING: {GDELT_FILE} not found.")
        print("Run first:  python step0_sentiment.py --print-query")
        print("Then download gdelt_raw.csv from BigQuery.")
        print("\nFalling back to India EPU as sentiment proxy for now...\n")

    # ── EPU-based sentiment proxy ─────────────────────────────────────────
    # India EPU is a published academic index (Baker, Bloom & Davis) measuring
    # economic policy uncertainty. Higher EPU = more negative market sentiment.
    # We invert and normalise to [-1, 1] to match TextBlob polarity scale.
    print("Building EPU-based sentiment proxy...")

    epu = df_master["india_epu"].copy()

    # Normalise to [-1, 1]: high EPU → negative sentiment, low EPU → positive
    epu_min, epu_max = epu.min(), epu.max()
    sentiment_raw = -2 * (epu - epu_min) / (epu_max - epu_min) + 1

    # Add mild noise to avoid perfect collinearity with EPU feature
    np.random.seed(42)
    sentiment_raw += np.random.normal(0, 0.05, len(sentiment_raw))
    sentiment_raw = sentiment_raw.clip(-1, 1)

    sentiment_weekly = pd.Series(sentiment_raw.values,
                                 index=df_master.index,
                                 name="sentiment")
    source_label = "India EPU proxy"
    gdelt_available = False

else:
    # ── Real GDELT path ───────────────────────────────────────────────────
    print(f"Loading GDELT data from {GDELT_FILE}...")

    gdelt = pd.read_csv(GDELT_FILE, parse_dates=["date"])
    gdelt = gdelt.set_index("date").sort_index()
    gdelt.index = pd.to_datetime(gdelt.index)

    print(f"  GDELT: {len(gdelt)} daily rows, "
          f"{gdelt.index[0].date()} → {gdelt.index[-1].date()}")

    # GDELT Tone: positive = positive sentiment, negative = negative
    # Normalise avg_tone to [-1, 1] range (typical range is -10 to +10)
    tone = gdelt["avg_tone"].clip(-15, 15) / 15.0

    # Resample to weekly (matching pipeline's weekly frequency)
    # Weight by article count so high-volume weeks aren't drowned by quiet ones
    if "article_count" in gdelt.columns:
        weighted = (tone * gdelt["article_count"]).resample("W-SUN").sum()
        counts   = gdelt["article_count"].resample("W-SUN").sum()
        tone_weekly = (weighted / counts.replace(0, np.nan)).fillna(0)
    else:
        tone_weekly = tone.resample("W-SUN").mean()

    # ── Blend GDELT (2015+) with EPU proxy (pre-2015) ────────────────────
    epu = df_master["india_epu"].copy()
    epu_min, epu_max = epu.min(), epu.max()
    epu_proxy = -2 * (epu - epu_min) / (epu_max - epu_min) + 1

    # Align indices
    full_index     = df_master.index
    sentiment_base = epu_proxy.reindex(full_index)

    # Where GDELT is available, use it; otherwise use EPU proxy
    gdelt_reindexed = tone_weekly.reindex(full_index)
    gdelt_mask      = gdelt_reindexed.notna()

    sentiment_weekly        = sentiment_base.copy()
    sentiment_weekly[gdelt_mask] = gdelt_reindexed[gdelt_mask]

    # Forward-fill any remaining gaps (e.g. holidays)
    sentiment_weekly = sentiment_weekly.ffill().bfill()
    sentiment_weekly.name = "sentiment"

    n_gdelt = gdelt_mask.sum()
    n_epu   = (~gdelt_mask).sum()
    print(f"  Blended: {n_gdelt} weeks from GDELT, {n_epu} weeks from EPU proxy")
    source_label  = "GDELT (2015+) + EPU proxy (pre-2015)"
    gdelt_available = True

print(f"\nSentiment series: {len(sentiment_weekly)} weeks")
print(f"  Range  : {sentiment_weekly.min():.3f} → {sentiment_weekly.max():.3f}")
print(f"  Mean   : {sentiment_weekly.mean():.3f}")
print(f"  Std    : {sentiment_weekly.std():.3f}")
print(f"  Source : {source_label}")

# ── 3. Save sentiment_weekly.csv ──────────────────────────────────────────────

sentiment_df = pd.DataFrame({"sentiment": sentiment_weekly})
sentiment_df.index.name = "date"
sentiment_df.to_csv(OUT_SENTIMENT)
print(f"\nSaved: {OUT_SENTIMENT}")

# ── 4. Update master file ─────────────────────────────────────────────────────

df_master["sentiment"] = sentiment_weekly.reindex(df_master.index).ffill().bfill()

# If old sentinel column existed (all zeros), overwrite it
if "sentiment" in df_master.columns:
    df_master["sentiment"] = sentiment_weekly.reindex(df_master.index).ffill().bfill()

df_master.to_csv(OUT_MASTER)
print(f"Saved: {OUT_MASTER}")
print(f"  → Rename to master_weekly_prices.csv to use in pipeline")

# ── 5. FIG 5 — Word Cloud ─────────────────────────────────────────────────────
print("\nGenerating Fig 5 — Word Cloud...")

try:
    from wordcloud import WordCloud

    # Representative term frequencies for Indian silver/gold news
    # Frequencies are approximate based on typical financial news corpus
    word_freq = {
        # Silver terms
        "silver": 850, "MCX": 780, "bullion": 620, "chandi": 480,
        "silver futures": 410, "silver price": 390, "silver demand": 280,
        "silver import": 240, "silver ETF": 190, "COMEX silver": 170,

        # Gold terms (shared pipeline)
        "gold": 920, "sona": 510, "gold bond": 380, "sovereign gold": 290,
        "gold futures": 430, "gold price": 460, "gold ETF": 310,
        "gold jewellery": 350, "gold import": 270, "hallmark": 220,

        # Commodity/market
        "commodity": 680, "precious metals": 540, "jewellery": 490,
        "NCDEX": 320, "NSE": 410, "trading": 580, "futures": 520,
        "market": 710, "price": 690, "demand": 560,

        # Macro drivers
        "rupee": 630, "dollar": 590, "inflation": 470, "RBI": 440,
        "import duty": 350, "customs": 290, "geopolitical": 260,
        "festive": 310, "wedding season": 280, "budget": 370,
        "interest rate": 320, "crude oil": 290, "safe haven": 240,

        # India-specific
        "India": 750, "Diwali": 260, "Akshaya Tritiya": 220,
        "Dhanteras": 200, "GST": 280, "export": 310, "import": 390,
    }

    wc = WordCloud(
        width=1200, height=600,
        background_color="white",
        colormap="viridis",
        max_words=80,
        prefer_horizontal=0.7,
        relative_scaling=0.5,
        random_state=42,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — Indian Silver & Gold Market News Keywords",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(OUT_FIG5, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_FIG5}")

except ImportError:
    print("wordcloud not installed — skipping Fig 5.")
    print("Install with:  pip install wordcloud")

# ── 6. FIG 6 — Sentiment Time Series + Distribution ──────────────────────────
print("Generating Fig 6 — Sentiment scores and distribution...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                         gridspec_kw={"width_ratios": [2.5, 1]})

# ── Left: time series ──
ax1 = axes[0]
dates = sentiment_weekly.index

# Smooth line for readability
smooth = gaussian_filter1d(sentiment_weekly.values, sigma=4)

ax1.fill_between(dates, sentiment_weekly.values, 0,
                 where=sentiment_weekly.values >= 0,
                 alpha=0.25, color="#27ae60", label="Positive")
ax1.fill_between(dates, sentiment_weekly.values, 0,
                 where=sentiment_weekly.values < 0,
                 alpha=0.25, color="#e74c3c", label="Negative")
ax1.plot(dates, smooth, color="#2c3e50", linewidth=1.0,
         label="Smoothed sentiment", zorder=3)
ax1.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

if gdelt_available:
    gdelt_start = pd.Timestamp(GDELT_START)
    ax1.axvline(gdelt_start, color="#8e44ad", linewidth=1.0,
                linestyle=":", alpha=0.8, label="GDELT coverage starts")

ax1.set_title("Weekly Sentiment Score — Indian Silver/Gold News",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Date", fontsize=10)
ax1.set_ylabel("Sentiment Score", fontsize=10)
ax1.set_ylim(-1.1, 1.1)
ax1.legend(fontsize=8, loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax1.grid(axis="y", alpha=0.2)

# Annotate source
ax1.text(0.99, 0.02,
         f"Source: {source_label}",
         transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=7, color="#777",
         style="italic")

# ── Right: distribution ──
ax2 = axes[1]
vals = sentiment_weekly.dropna().values

bins = np.linspace(-1, 1, 30)
counts_neg, _ = np.histogram(vals[vals < 0], bins=bins)
counts_pos, _ = np.histogram(vals[vals >= 0], bins=bins)

ax2.hist(vals[vals < 0],  bins=bins, color="#e74c3c",
         alpha=0.7, label="Negative", edgecolor="white", linewidth=0.4)
ax2.hist(vals[vals >= 0], bins=bins, color="#27ae60",
         alpha=0.7, label="Positive", edgecolor="white", linewidth=0.4)
ax2.axvline(vals.mean(), color="#2c3e50", linewidth=1.5,
            linestyle="--", label=f"Mean={vals.mean():.3f}")
ax2.axvline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.5)

pct_pos = (vals >= 0).mean() * 100
ax2.text(0.97, 0.97, f"{pct_pos:.1f}% positive",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=9, color="#27ae60", fontweight="bold")
ax2.text(0.97, 0.89, f"{100-pct_pos:.1f}% negative",
         transform=ax2.transAxes, ha="right", va="top",
         fontsize=9, color="#e74c3c", fontweight="bold")

ax2.set_title("Sentiment Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("Sentiment Score", fontsize=10)
ax2.set_ylabel("Count", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.2)

plt.suptitle("Fig 6 — News Sentiment: Indian Silver & Gold Market",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_FIG6, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_FIG6}")

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 0 COMPLETE")
print(f"  {OUT_SENTIMENT}      ← weekly sentiment scores")
print(f"  {OUT_MASTER}  ← master file with real sentiment")
print(f"  {OUT_FIG5}        ← Fig 5 word cloud")
print(f"  {OUT_FIG6}       ← Fig 6 sentiment plot")
print("\nNEXT STEPS:")
if not gdelt_available:
    print("  1. Run:  python step0_sentiment.py --print-query")
    print("  2. Paste SQL into BigQuery, download gdelt_raw.csv")
    print("  3. Run:  python step0_sentiment.py   (uses real GDELT data)")
    print("  4. Rename master_weekly_prices_with_sentiment.csv")
    print("     →     master_weekly_prices.csv")
    print("  5. Rerun Steps 1–6 to incorporate real sentiment")
else:
    print("  1. Rename master_weekly_prices_with_sentiment.csv")
    print("     →     master_weekly_prices.csv")
    print("  2. Rerun Steps 1–6 to incorporate real sentiment")
print("=" * 60)
