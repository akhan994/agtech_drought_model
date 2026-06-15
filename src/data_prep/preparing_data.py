"""
This file will be used to prepare the data for the drought classifier.

What we need  from this is:

build the label (the y): take the cumulative USDM percent-area columns --> convert them in new per-category area --> set the argmax as the one that defines that area --> one  of 6 classes --> one-hot 

build the sliding window: for each USDM week t, gather the prior input_window weeks of weather features as X, and the class at week t + leadtime as y

merge weather (NOAA precipitation/NDVI data) on the weekly USDM grid

it will have one function that returns x_train, y_train, x_val, y_val,l x_test, y_test and datees and an optional scaler """