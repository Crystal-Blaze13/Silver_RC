import pandas as pd
import os

DATA_DIR = "financial_data"

dfs = {}

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        name = file.replace(".csv","")
        df = pd.read_csv(os.path.join(DATA_DIR, file), parse_dates=["date"])
        df = df.rename(columns={"close": name})
        dfs[name] = df

# Merge all on date using inner join
merged = dfs["silver"]

for name in dfs:
    if name != "silver":
        merged = merged.merge(dfs[name], on="date", how="inner")

# Sort by date
merged = merged.sort_values("date").reset_index(drop=True)

print("Final merged shape:", merged.shape)
print(merged.head())
print(merged.tail())