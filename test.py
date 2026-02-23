"""
FULL PIPELINE (WEEKLY) — Prices + Google Trends (weekly) → merged weekly log-returns dataset

Inputs (files in current folder):
- silver.csv
- gold.csv
- brent_crude.csv
- dxy.csv
- sp500.csv
- rc google trends data.csv

What it does:
1) Loads + cleans Google Trends weekly (DD-MM-YYYY), removes duplicates
2) Loads each price CSV, auto-detects Date + Close/Adj Close column
3) Resamples prices to weekly (W-SUN) to match trends
4) Computes log returns for all price series
5) Merges everything on weekly dates (inner join)
6) Creates trends log-diff feature (weekly)
7) Saves final dataset: merged_weekly_dataset.csv
"""

import pandas as pd
import numpy as np

# -----------------------------
# Helpers
# -----------------------------
def load_trends_weekly(path: str) -> pd.DataFrame:
    trends = pd.read_csv(path)
    if trends.shape[1] < 2:
        raise ValueError("Trends CSV must have at least 2 columns: Time and value.")

    # Expect columns like ['Time', 'silver price']
    date_col = trends.columns[0]
    val_col  = trends.columns[1]

    trends = trends.rename(columns={date_col: "Date", val_col: "trends_raw"}).copy()

    # Parse DD-MM-YYYY (your file is like 28-12-2014)
    trends["Date"] = pd.to_datetime(trends["Date"], format="%d-%m-%Y", errors="coerce")
    trends["trends_raw"] = pd.to_numeric(trends["trends_raw"], errors="coerce")
    trends = trends.dropna(subset=["Date", "trends_raw"]).copy()

    trends = trends.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")
    trends = trends.set_index("Date").sort_index()

    # Sanity check: should mostly be 7-day steps
    deltas = trends.index.to_series().diff().dropna().value_counts().head(5)
    print("\n[Trends] top time deltas:\n", deltas)

    return trends[["trends_raw"]]


def load_price_series(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Find a date column (common: Date, date, time, Time)
    possible_date_cols = [c for c in df.columns if c.lower() in ["date", "time", "datetime"]]
    if not possible_date_cols:
        # fallback: assume first column is date-like
        date_col = df.columns[0]
    else:
        date_col = possible_date_cols[0]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.sort_values(date_col).set_index(date_col)

    # Choose price column
    # Prefer Adj Close if present; else Close; else any numeric column
    candidates = []
    for col in df.columns:
        if col.lower() in ["adj close", "adj_close", "adjclose"]:
            candidates = [col]
            break
    if not candidates:
        for col in df.columns:
            if col.lower() == "close":
                candidates = [col]
                break
    if not candidates:
        # fallback: first numeric column
        numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if not numeric_cols:
            raise ValueError(f"No numeric price column found in {path}.")
        candidates = [numeric_cols[0]]

    price_col = candidates[0]
    s = pd.to_numeric(df[price_col], errors="coerce").rename(name).to_frame()
    s = s.dropna()

    return s


def to_weekly_last(df: pd.DataFrame) -> pd.DataFrame:
    # Weekly on Sunday to match your trends dates (e.g., 2015-01-04 is a Sunday)
    return df.resample("W-SUN").last()


def log_return(series: pd.Series) -> pd.Series:
    # log(P_t/P_{t-1})
    return np.log(series / series.shift(1))


# -----------------------------
# 1) Load Trends (weekly)
# -----------------------------
trends = load_trends_weekly("rc google trends data.csv")
print("[Trends] shape:", trends.shape)
print(trends.head())

# Create trends weekly log-diff feature (attention "return")
# Use log1p to be extra safe if small values appear; your min=22 so regular log is fine too.
trends["trends_log1p"] = np.log1p(trends["trends_raw"])
trends["trends_r"] = trends["trends_log1p"].diff()

# Keep only needed
trends_feat = trends[["trends_raw", "trends_r"]].copy()

# -----------------------------
# 2) Load Prices (daily) and resample to weekly
# -----------------------------
silver = to_weekly_last(load_price_series("financial_data/silver.csv", "silver_px"))
gold  = to_weekly_last(load_price_series("financial_data/gold.csv", "gold_px"))
brent = to_weekly_last(load_price_series("financial_data/brent_crude.csv", "brent_px"))
dxy   = to_weekly_last(load_price_series("financial_data/dxy.csv", "dxy_px"))
sp500 = to_weekly_last(load_price_series("financial_data/sp500.csv", "sp500_px"))

print("\n[Prices] weekly shapes:",
      "silver", silver.shape,
      "gold", gold.shape,
      "brent", brent.shape,
      "dxy", dxy.shape,
      "sp500", sp500.shape)

# -----------------------------
# 3) Compute weekly log returns for prices
# -----------------------------
silver["silver_r"] = log_return(silver["silver_px"])
gold["gold_r"]     = log_return(gold["gold_px"])
brent["brent_r"]   = log_return(brent["brent_px"])
dxy["dxy_r"]       = log_return(dxy["dxy_px"])
sp500["sp500_r"]   = log_return(sp500["sp500_px"])

# Keep only return columns (you can keep px columns too if you want)
silver_r = silver[["silver_r"]]
gold_r   = gold[["gold_r"]]
brent_r  = brent[["brent_r"]]
dxy_r    = dxy[["dxy_r"]]
sp500_r  = sp500[["sp500_r"]]

# -----------------------------
# 4) Merge all (inner join on dates)
# -----------------------------
merged = (
    silver_r
    .join(gold_r, how="inner")
    .join(brent_r, how="inner")
    .join(dxy_r, how="inner")
    .join(sp500_r, how="inner")
    .join(trends_feat, how="inner")
)

# Drop initial NaNs from returns/diffs
merged = merged.dropna().copy()

# -----------------------------
# 5) Quick diagnostics
# -----------------------------
print("\n[Merged] shape:", merged.shape)
print(merged.head())
print("\n[Merged] date range:", merged.index.min(), "→", merged.index.max())
print("\nMissing per column:\n", merged.isna().sum())
print("\nDescribe:\n", merged.describe())

# -----------------------------
# 6) Save
# -----------------------------
OUT = "merged_weekly_dataset.csv"
merged.to_csv(OUT)
print("\nSaved:", OUT)