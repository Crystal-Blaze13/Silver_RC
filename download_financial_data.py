import yfinance as yf
import pandas as pd
import os

# Configuration
START_DATE = "2016-01-01"
END_DATE = "2026-02-20"
OUTPUT_DIR = "financial_data"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tickers to download
tickers = {
    "silver":    "SI=F",
    "brent_crude": "BZ=F",
    "sp500":     "^GSPC",
    "dxy":       "DX-Y.NYB",
    "gold":      "GC=F",
}

for name, ticker in tickers.items():
    print(f"Downloading {name} ({ticker})...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    if df.empty:
        print(f"  ⚠️  No data returned for {ticker}")
        continue

    # Keep only Close price + flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"

    filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(filepath)
    print(f"  ✅ Saved {len(df)} rows → {filepath}")

print("\nAll done! CSVs saved to:", os.path.abspath(OUTPUT_DIR))