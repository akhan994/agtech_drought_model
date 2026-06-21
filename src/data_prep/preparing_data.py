"""
This file will be used to prepare the data for the drought classifier.

What we need  from this is:

build the label (the y): take the cumulative USDM percent-area columns --> convert them in new per-category area --> set the argmax as the one that defines that area --> one  of 6 classes --> one-hot 

build the sliding window: for each USDM week t, gather the prior input_window weeks of weather features as X, and the class at week t + leadtime as y

merge weather (NOAA precipitation/NDVI data) on the weekly USDM grid

it will have one function that returns x_train, y_train, x_val, y_val,l x_test, y_test and datees and an optional scaler 

dataset_path, input_window, leadtime, rotation, scale, model, verbose
"""

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# the only NOAA station with complete temperature over the full 2000-2026 span
TEMP_STATION = "USC00414782"


def _bin_to_weeks(daily, value_cols, agg, week_grid):
    """Map each raw observation date to the USDM week it falls in, then aggregate.

    daily     : DataFrame with a 'week_start' column holding the RAW obs dates
                (daily for NOAA, 16-day for NDVI) plus value_cols.
    week_grid : DataFrame with a single 'week_start' column = the USDM weeks.
    returns   : one row per USDM week -> week_start + aggregated value_cols.
    """
    g = week_grid.rename(columns={"week_start": "wk"}).sort_values("wk")
    binned = pd.merge_asof(
        daily.sort_values("week_start"),
        g,
        left_on="week_start", right_on="wk",
        direction="backward",            # assign each obs to the week that started on/before it
        tolerance=pd.Timedelta(days=6),  # ...but only within that 7-day window
    ).dropna(subset=["wk"])
    weekly = binned.groupby("wk")[value_cols].agg(agg)
    weekly.index.name = "week_start"
    return weekly.reset_index()

def preparing_data(data_path, 
                   feature_cols, 
                   input_window, 
                   leadtime, 
                   label_col="class_number", 
                   split=(0.7, 0.15, 0.15), 
                   scale=False, 
                   num_classes=6, 
                   verbose=0):

    drought_df = pd.read_csv(r"raw_data\USDM\USDM_labels.csv")
    ndvi_df = pd.read_csv(r"raw_data/google_earth_engine/NDVI For Kerr County (2000-2024).csv")
    noaa_df = pd.read_csv(r"raw_data\NOAA\NOAA_combined_2000-2026.csv")

    ndvi_df.rename(columns = {'date': 'week_start'}, inplace=True)
    noaa_df.rename(columns = {'DATE': 'week_start'}, inplace=True)

    # each source has its OWN string format, so pair each df with its format and
    # parse individually - you can't share one format= across all three.
    # (the date columns were renamed to 'week_start' just above.)
    formats = [(drought_df, "%m/%d/%y"), (ndvi_df, "%m-%d-%y"), (noaa_df, "%m-%d-%y")]
    for df, fmt in formats:
        df["week_start"] = pd.to_datetime(df["week_start"], format=fmt)  # assign back; no inplace= on to_datetime
        df.sort_values(by="week_start", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ---- resample every source onto the weekly USDM grid -------------------
    # NOAA precip/temp must be numeric; coerce so blanks/'T' become NaN, not strings
    noaa_df[["PRCP", "TMAX", "TMIN"]] = noaa_df[["PRCP", "TMAX", "TMIN"]].apply(pd.to_numeric, errors="coerce")

    # the USDM weeks ARE the target grid; every other source is resampled onto them
    weeks = drought_df[["week_start"]]

    # NOAA daily -> weekly:
    #   precip = average across all reporting stations each day, then weekly TOTAL
    daily_precip = noaa_df.groupby("week_start", as_index=False)["PRCP"].mean()
    weekly_precip = _bin_to_weeks(daily_precip, ["PRCP"], "sum", weeks).rename(columns={"PRCP": "prcp"})
    #   temp = the single full-coverage station, weekly MEAN of TMAX/TMIN
    temp_df = noaa_df[noaa_df["STATION"] == TEMP_STATION]
    daily_temp = temp_df.groupby("week_start", as_index=False)[["TMAX", "TMIN"]].mean()
    weekly_temp = _bin_to_weeks(daily_temp, ["TMAX", "TMIN"], "mean", weeks).rename(columns={"TMAX": "tmax", "TMIN": "tmin"})

    # NDVI 16-day -> weekly MEAN (sparse; interpolated below)
    weekly_ndvi = _bin_to_weeks(ndvi_df[["week_start", "NDVI"]], ["NDVI"], "mean", weeks).rename(columns={"NDVI": "ndvi"})

    # merge everything onto the weekly USDM grid -> one master table
    master = (drought_df.merge(weekly_precip, on="week_start", how="left")
                        .merge(weekly_temp, on="week_start", how="left")
                        .merge(weekly_ndvi, on="week_start", how="left"))

    # cut to the NDVI-covered span (NDVI bottlenecks us at ~2024) so we never extrapolate it
    master = master[master["week_start"] <= ndvi_df["week_start"].max()].reset_index(drop=True)

    # keep the grid CONTIGUOUS (positional windowing depends on it): fill gaps
    weather_cols = ["prcp", "tmax", "tmin", "ndvi"]
    master[weather_cols] = master[weather_cols].interpolate(limit_direction="both")

    # contiguity + completeness sanity checks
    assert (master["week_start"].diff().dropna() == pd.Timedelta(days=7)).all(), "weekly grid is not contiguous!"
    assert master[weather_cols].isna().sum().sum() == 0, "features still contain NaN after fill!"

    if verbose:
        print(f"[master] {len(master)} weeks "
              f"{master['week_start'].min().date()} -> {master['week_start'].max().date()}")
        print(master[weather_cols].describe().round(3))

    # separate features / labels / dates  (now from the merged master)

    features = master[feature_cols].to_numpy()  # shape (N, F)
    labels = master[label_col].to_numpy()       # shape (N,)
    dates = master["week_start"].to_numpy()      # shape (N,)

    N = len(master)
    F = len(feature_cols)
    W, L = input_window, leadtime

    # build windows
    # anchor week  i = the last week that goes into the input
    X_list, y_list, target_dates = [], [], []

    for i in range(W-1, N-L):
        window = features[i-W+1 : i+1] # (W, F)
        X_list.append(window.flatten()) # -> (W*F) flatten bc MLP
        y_list.append(labels[i+L]) # single cllass integer
        target_dates.append(dates[i+L])

    X = np.array(X_list) # (num_samples, W*F)
    y_int = np.array(y_list) # (num_samples)
    target_dates = np.array(target_dates)

    # one-hot the label
    y = to_categorical(y_int, num_classes=num_classes)

    # chronological split (can't shuffle)
    n = len(X)
    i_train = int(n * split[0])
    i_val = int(n * (split[0] + split[1]))

    # slice each array at the split boundaries (chronological order preserved)
    x_train, y_train, train_dates = X[:i_train], y[:i_train], target_dates[:i_train]
    x_val, y_val, val_dates = X[i_train:i_val], y[i_train:i_val], target_dates[i_train:i_val]
    x_test, y_test, test_dates = X[i_val:], y[i_val:], target_dates[i_val:]

    # scale features (fit on TRAIN only; never scale the one-hot y)
    scaler = None
    if scale:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    if verbose:
        print(f"[split] train={len(x_train)} val={len(x_val)} test={len(x_test)} | "
              f"train<= {pd.Timestamp(train_dates.max()).date()}  test>= {pd.Timestamp(test_dates.min()).date()}")

    return (x_train, y_train, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, scaler)

if __name__ == "__main__":
    out = preparing_data(
        data_path=None,
        feature_cols=["prcp", "tmax", "tmin", "ndvi"],
        input_window=4,
        leadtime=4,
        scale=True,
        verbose=1,
    )
    x_train, y_train, x_val, y_val, x_test, y_test, *_, scaler = out
    print("\nshapes:")
    print("  x_train", x_train.shape, "y_train", y_train.shape)
    print("  x_val  ", x_val.shape, "y_val  ", y_val.shape)
    print("  x_test ", x_test.shape, "y_test ", y_test.shape)