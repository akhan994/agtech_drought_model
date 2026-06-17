"""
Combine the four NOAA daily-summary CSVs into a single file.

- Concatenates every station row across the 4 time-range files.
- The files have inconsistent columns (the 2000-2005 file has no AWND; the
  2005-2010 file has no NAME); pandas aligns them by column name and fills the
  gaps with NaN.
- Dedups the boundary-year overlaps (e.g. 2005-01-01 appears in two files) on
  (STATION, DATE), keeping the first occurrence.
- Sorts DESCENDING by date (newest first) to match the drought.gov file's order;
  ties within a date are ordered by STATION.
- Original DATE strings (MM-DD-YY) are preserved as-is; a parsed copy is used
  only for sorting, then dropped.

Output: raw_data/NOAA/NOAA_combined_2000-2026.csv  (inputs left untouched)
"""

import glob
import pandas as pd

SRC_GLOB = "raw_data/NOAA/NOAA Daily Summaries*.csv"
OUT = "raw_data/NOAA/NOAA_combined_2000-2026.csv"

files = sorted(glob.glob(SRC_GLOB))
df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)

before = len(df)
df = df.drop_duplicates(subset=["STATION", "DATE"], keep="first")
deduped = before - len(df)

# parsed key for sorting only (descending by date, then station)
key = pd.to_datetime(df["DATE"], format="%m-%d-%y")
df = (df.assign(_k=key)
        .sort_values(["_k", "STATION"], ascending=[False, True])
        .drop(columns="_k"))

# stable, explicit column order (union across all files)
col_order = ["STATION", "NAME", "DATE", "PRCP", "TMAX", "TMIN", "AWND"]
df = df.reindex(columns=[c for c in col_order if c in df.columns])

df.to_csv(OUT, index=False)

print(f"combined {len(files)} files: {before} -> {len(df)} rows ({deduped} overlap dups removed)")
print(f"stations: {df['STATION'].nunique()}  |  wrote -> {OUT}")
print("\nhead:\n", df.head().to_string(index=False))
print("\ntail:\n", df.tail().to_string(index=False))
