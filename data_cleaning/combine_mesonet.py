"""
Combine the seven TxMesonet timeseries CSVs (station TWB44) into a single file.

- Concatenates all 7 (~April-to-April) files.
- Strips leading/trailing whitespace from the column names and drops the empty
  trailing column produced by the source file's trailing comma.
- Dedups boundary overlaps on (Station_ID, Date_Time (UTC)).
- Sorts DESCENDING by Date_Time (newest first) to match the drought.gov order.
- Date_Time is already ISO-8601 UTC; preserved as-is.

Output: raw_data/TXMesonet/TXMesonet_combined_2019-2026.csv  (inputs untouched)
"""

import glob
import pandas as pd

SRC_GLOB = "raw_data/TXMesonet/TxMeso Timeseries*.csv"
OUT = "raw_data/TXMesonet/TXMesonet_combined_2019-2026.csv"
DATE_COL = "Date_Time (UTC)"

files = sorted(glob.glob(SRC_GLOB))
frames = []
for f in files:
    d = pd.read_csv(f, dtype=str)
    d.columns = d.columns.str.strip()
    d = d.loc[:, [c for c in d.columns if c != ""]]   # drop empty trailing column
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

before = len(df)
df = df.drop_duplicates(subset=["Station_ID", DATE_COL], keep="first")
deduped = before - len(df)

key = pd.to_datetime(df[DATE_COL])
df = df.assign(_k=key).sort_values("_k", ascending=False).drop(columns="_k")

df.to_csv(OUT, index=False)

print(f"combined {len(files)} files: {before} -> {len(df)} rows ({deduped} overlap dups removed)")
print(f"date range: {key.min()} -> {key.max()}  |  wrote -> {OUT}")
print("\ncolumns:", list(df.columns))
print("\nhead:\n", df.head(3).to_string(index=False))
