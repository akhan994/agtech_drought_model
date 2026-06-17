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
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

def preparing_data(data_path, feature_cols, input_window, leadtime, label_col="class_number", split=(0.7, 0.15, 0.15), scale=False, num_classes=6, verbose=0):

    drought_df = pd.read_csv(r"raw_data\USDM\USDM_labels.csv")
    ndvi_df = pd.read_csv(r"raw_data/google_earth_engine/NDVI For Kerr County (2000-2024).csv")
    noaa_df = pd.read_csv(r"raw_data\NOAA\NOAA_combined_2000-2026.csv")
    
    dfs = drought_df, ndvi_df, noaa_df

    ndvi_df.rename(columns = {'date': 'week_start'})
    noaa_df.rename(columns = {'DATE': 'week_start'})

    for df in dfs:
        pd.to_datetime(df['week_start'], format = '%m/%d/%y', inplace=True)
        df.sort_values(by='week_start', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["week_start"].diff().dropna() == pd.Timedelta(days=7)

        # things i need to do:
        # i'm going to cut the max year at 2024 since ndvi is bottlenecking us there
        # i don't think i've sampled the dataframes to a weekly version yet 
        # for the noaa csv's precipitation, i need to get the weekly average from the data
        # and for the noaa csv's temp, i need to extract the one station that is active through the entire time range and use that as my temp data
        # i can append the columns of the week_start, prcp, temp, drought label, and ndvi into one dataframe, so that it's easier to extract the feature_cols and the one singular label_col
        # also for visualizing purposes, i can take this one dataframe to visualize the observed data across the years in a plotly plot (TODO LATER)

    
    # separate features / labels / dates

    features = df[feature_cols].to_numpy()  # shape (N, F)
    labels = df[label_col].to_numpy()       # shape (N,)
    dates = df["week_start"].to_numpy()     # shape (N,)

    N = len(df)
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
        pass

    X = np.array(X_list) # (num_samples, W*F)
    y_int = np.array(y_list) # (num_samples)
    target_dates = np.arrayy(target_dates)

    # one-hot the label
    y = to_categorical(y_int, num_classes=num_classes)

    # chronological split (can't shuffle)
    n = len(X)
    i_train = int(n * split[0])
    i_val = int(n * (split[0] + split[1]))
    
    # scale features 
    scaler = None
    if scale:
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

    return (x_train, y_train, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, scaler)

if __name__ == "__main__":
     preparing_data()