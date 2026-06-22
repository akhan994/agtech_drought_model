"""
Combine the NOAA daily-summary CSVs into a single file.

Unit fix (2026-06-17): the original "NOAA Daily Summaries (2005-2010).csv" was
exported in GHCND METRIC units (temp in tenths of degC, precip in tenths of mm),
which produced impossible values (tmax 380, precip 1572). It is EXCLUDED here and
replaced by "noaa new data 2005 - 2009.csv", re-pulled in standard units
(degF / inches). The new file ends 2009-12-31 and the 2010-2019 file starts
2010-01-01, so coverage stays continuous.

- Concatenates every station row across the time-range files.
- Files have inconsistent columns; pandas aligns by name and fills gaps with NaN.
- The replacement file uses ISO dates + extra columns; normalized to the common
  schema (STATION, NAME, DATE[MM-DD-YY], PRCP, TMAX, TMIN).
- Dedups boundary overlaps on (STATION, DATE), keeping the first occurrence.
- Sorts DESCENDING by date (newest first) to match the drought.gov file's order.

Output: raw_data/NOAA/NOAA_combined_2000-2026.csv  (inputs left untouched)
"""

import glob
import pandas as pd

SRC_GLOB = "raw_data/NOAA/NOAA Daily Summaries*.csv"
METRIC_FILE = "2005-2010"                                      # substring of the bad file to drop
REPLACEMENT = "raw_data/NOAA/noaa new data 2005 - 2009.csv"    # standard-unit re-pull
OUT = "raw_data/NOAA/NOAA_combined_2000-2026.csv"
COMMON = ["STATION", "NAME", "DATE", "PRCP", "TMAX", "TMIN"]

# original-format files, minus the metric one
files = [f for f in sorted(glob.glob(SRC_GLOB)) if METRIC_FILE not in f]
frames = [pd.read_csv(f, dtype=str) for f in files]

# replacement file: keep common columns, convert ISO date -> MM-DD-YY to match
new = pd.read_csv(REPLACEMENT, dtype=str)[COMMON].copy()
new["DATE"] = pd.to_datetime(new["DATE"]).dt.strftime("%m-%d-%y")
frames.append(new)

df = pd.concat(frames, ignore_index=True)

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

print(f"combined {len(files)} original + 1 replacement file: {before} -> {len(df)} rows "
      f"({deduped} overlap dups removed)")
print(f"stations: {df['STATION'].nunique()}  |  wrote -> {OUT}")

# quick unit sanity check on the temp station across the formerly-metric years
chk = df[df["STATION"] == "USC00414782"].copy()
chk["TMAX"] = pd.to_numeric(chk["TMAX"], errors="coerce")
chk["yr"] = pd.to_datetime(chk["DATE"], format="%m-%d-%y").dt.year
print("\ntemp-station TMAX yearly mean 2004-2010 (all should be ~75-82 degF):")
print(chk[chk["yr"].between(2004, 2010)].groupby("yr")["TMAX"].mean().round(1).to_string())
