"""
Data preparation for the drought classifier

Pipeline:
  1. build_weekly_master(): merge the three raw sources onto the weekly USDM grid
       - USDM labels  (weekly)         -> class_number / class_label  (already relabeled)
       - NOAA daily   -> weekly precip + weekly temp
       - NDVI 16-day  -> weekly NDVI
  2. preparing_data(): slide a window over the weekly master, one-hot the label,
       split chronologically, optionally scale, and return numpy arrays.

DECISIONS MADE HERE (flagged so they can be revisited):
  - precip feature = weekly TOTAL of the daily cross-station MEAN precip
        (drought is about accumulated rainfall; switch agg to "mean" for a rate).
  - temp features  = weekly MEAN of TMAX/TMIN from the single station with full
        coverage (USC00414782 = KERRVILLE 3 NNE); the rest of NOAA is precip-only.
  - NDVI (16-day) is linearly interpolated onto the weekly grid.
  - residual feature gaps are linearly interpolated, edges back/forward filled,
        so the weekly grid stays CONTIGUOUS (positional windowing depends on this).
  - series cut to <= last NDVI date (~2024), since NDVI ends 2024.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

USDM_PATH = r"raw_data/USDM/USDM_labels.csv"
NOAA_PATH = r"raw_data/NOAA/NOAA_combined_2000-2026.csv"
NDVI_PATH = r"raw_data/google_earth_engine/NDVI For Kerr County (2000-2024).csv"
SOIL_PATH = r"raw_data/" #fill in later

# the only NOAA station with complete temperature over the full 2000-2026 span
TEMP_STATION = "USC00414782"

def _weekly(daily, value_cols, agg, weeks):
    """Bin daily rows into the USDM weeks they fall in, then aggregate.
    pd.merge_asof will merge according to the nearest key. 

    daily : DataFrame with a 'date' column + value_cols
    weeks : DataFrame with a single 'week_start' column (sorted ascending)
    """
    binned = pd.merge_asof(
        daily.sort_values("date"),
        weeks,
        left_on="date",
        right_on="week_start",
        direction="backward",            # map each date to the week that started on/before it
        tolerance=pd.Timedelta(days=6),  # ...but only within that 7-day window
    ).dropna(subset=["week_start"])
    return binned.groupby("week_start")[value_cols].agg(agg)

def build_weekly_master(verbose=0, save_path=None):
    """Return one weekly dataframe: week_start, prcp, tmax, tmin, ndvi, class_number, class_label."""

    # ---- USDM labels = the weekly grid -------------------------------------
    usdm = pd.read_csv(USDM_PATH)
    usdm["week_start"] = pd.to_datetime(usdm["week_start"], format="%m/%d/%y")
    usdm = usdm.sort_values("week_start").reset_index(drop=True)
    weeks = usdm[["week_start"]].copy()

    # ---- NOAA (daily -> weekly) --------------------------------------------
    noaa = pd.read_csv(NOAA_PATH)
    noaa["date"] = pd.to_datetime(noaa["DATE"], format="%m-%d-%y")
    for c in ("PRCP", "TMAX", "TMIN"):
        noaa[c] = pd.to_numeric(noaa[c], errors="coerce")  # 'T'/blank -> NaN

    # precip: average across all reporting stations per day, then weekly total
    daily_precip = (noaa.groupby("date", as_index=False)["PRCP"].mean()
                        .rename(columns={"PRCP": "prcp"}))
    weekly_precip = _weekly(daily_precip, ["prcp"], "sum", weeks)

    # temp: only the full-coverage station, weekly mean of daily TMAX/TMIN
    temp = noaa[noaa["STATION"] == TEMP_STATION]
    daily_temp = (temp.groupby("date", as_index=False)[["TMAX", "TMIN"]].mean()
                      .rename(columns={"TMAX": "tmax", "TMIN": "tmin"}))
    weekly_temp = _weekly(daily_temp, ["tmax", "tmin"], "mean", weeks)

    # ---- NDVI (16-day -> weekly) -------------------------------------------
    ndvi = pd.read_csv(NDVI_PATH)
    ndvi["date"] = pd.to_datetime(ndvi["date"], format="%m-%d-%y")
    ndvi = ndvi[["date", "NDVI"]].rename(columns={"NDVI": "ndvi"})
    weekly_ndvi = _weekly(ndvi, ["ndvi"], "mean", weeks)

    # ---- SOIL --------------------------------------------------------------
    # insert here

    # ---- merge everything onto the USDM weekly grid ------------------------
    master_df = (usdm.merge(weekly_precip, on="week_start", how="left")
                  .merge(weekly_temp, on="week_start", how="left")
                  .merge(weekly_ndvi, on="week_start", how="left"))

    # ---- past drought state as a usable INPUT feature ----------------------
    # The USDM class is highly predictive of future drought (a "persistence" signal).
    # It is safe to use as a feature because windowing only ever puts PAST weeks (<= t)
    # into X while the target is the class at t+leadtime (no leakage as long as L >= 1).
    # NOTE on encoding: class_number sets no_drought=5 (i.e. ABOVE D4), which is NOT
    # monotonic in severity. Fed raw to a model as a number, that ordering misleads it.
    # So we remap to an ordered 0..5 scale for use as a feature. class_number itself
    # stays untouched as the label/target.
    #   put "usdm_severity" in feature_cols to use it.
    master_df["usdm_severity"] = master_df["class_number"].map(
        {5: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5})

    # cut to the NDVI-covered span (NDVI ends 2024) so we never extrapolate it
    master_df = master_df[master_df["week_start"] <= ndvi["date"].max()].reset_index(drop=True)

    # keep the grid contiguous: interpolate interior gaps, fill the edges
    feature_cols = ["prcp", "tmax", "tmin", "ndvi"]
    master_df[feature_cols] = (master_df[feature_cols]
                            .interpolate(method="linear", limit_direction="both"))

    # contiguity check — positional windowing relies on a clean 7-day grid
    gaps = master_df["week_start"].diff().dropna()
    assert (gaps == pd.Timedelta(days=7)).all(), "weekly grid is not contiguous!"
    assert master_df[feature_cols].isna().sum().sum() == 0, "features still contain NaN!"

    if verbose:
        print(f"[master] {len(master_df)} weeks "
              f"{master_df['week_start'].min().date()} -> {master_df['week_start'].max().date()}")
        print(master_df[feature_cols].describe().round(3))

    if save_path:
        master_df.to_csv(save_path, index=False)
        if verbose:
            print(f"[master] saved -> {save_path}")

    return master_df


def preparing_data(data_path=None,
                   feature_cols=("prcp", "tmax", "tmin", "ndvi"),
                   input_window=4,
                   leadtime=4,
                   label_col="class_number",
                   split=(0.7, 0.15, 0.15),
                   scale=False,
                   num_classes=6,
                   verbose=0):
    """Slide a window over the weekly master and return train/val/test arrays.

    Returns (always 10 items; scaler is None when scale=False):
        x_train, y_train, x_val, y_val, x_test, y_test,
        train_dates, val_dates, test_dates, scaler
    """
    feature_cols = list(feature_cols)

    # if a prebuilt master CSV is passed, use it; otherwise build from raw sources
    if data_path:
        master = pd.read_csv(data_path, parse_dates=["week_start"])
    else:
        master = build_weekly_master(verbose=verbose)

    features = master[feature_cols].to_numpy()      # (N, F)
    labels = master[label_col].to_numpy()           # (N,)
    dates = master["week_start"].to_numpy()         # (N,)

    N = len(master)
    F = len(feature_cols)
    W, L = input_window, leadtime

    # ---- windowing ---------------------------------------------------------
    # anchor i = last week in the input; inputs [i-W+1 .. i], target at i+L
    X_list, y_list, target_dates = [], [], []
    for i in range(W - 1, N - L):
        window = features[i - W + 1: i + 1]    # (W, F)
        X_list.append(window.flatten())        # (W*F,)  flatten for the MLP
        y_list.append(labels[i + L])           # single class integer
        target_dates.append(dates[i + L])

    X = np.array(X_list)                       # (num_samples, W*F)
    y_int = np.array(y_list)                   # (num_samples,)
    target_dates = np.array(target_dates)

    if verbose:
        print(f"[windows] N={N}, W={W}, L={L} -> {len(X)} samples, "
              f"X shape {X.shape}")

    # ---- one-hot the label -------------------------------------------------
    y = np.eye(num_classes, dtype="float32")[y_int]   # (num_samples, num_classes)

    # ---- chronological split (NO shuffle) ----------------------------------
    n = len(X)
    i_train = int(n * split[0])
    i_val = int(n * (split[0] + split[1]))

    x_train, y_train, train_dates = X[:i_train], y[:i_train], target_dates[:i_train]
    x_val, y_val, val_dates = X[i_train:i_val], y[i_train:i_val], target_dates[i_train:i_val]
    x_test, y_test, test_dates = X[i_val:], y[i_val:], target_dates[i_val:]

    # ---- scale features (fit on TRAIN only) --------------------------------
    scaler = None
    if scale:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    if verbose:
        print(f"[split] train={len(x_train)} val={len(x_val)} test={len(x_test)}")
        print(f"[split] train<= {pd.Timestamp(train_dates.max()).date()} | "
              f"test>= {pd.Timestamp(test_dates.min()).date()}")  # should not overlap

    return (x_train, y_train, x_val, y_val, x_test, y_test,
            train_dates, val_dates, test_dates, scaler)


if __name__ == "__main__":
    # quick smoke test
    out = preparing_data(scale=True, verbose=1)
    x_train, y_train, x_val, y_val, x_test, y_test, *_ , scaler = out
    print("\nshapes:")
    print("  x_train", x_train.shape, "y_train", y_train.shape)
    print("  x_val  ", x_val.shape, "y_val  ", y_val.shape)
    print("  x_test ", x_test.shape, "y_test ", y_test.shape)
    print("  one-hot row sums all == 1:", bool((y_train.sum(axis=1) == 1).all()))
