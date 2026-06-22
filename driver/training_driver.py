# this driver will train the models 

# analogous to operational_mse_crps_driver.py from uq4ml_watertemp'

"""

The purpose of this script is to train a drought classification model. 

"""

from pathlib import Path

import numpy as np 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep.preparing_data import preparing_data

def build_model(input_dim, args):
    model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(args.neurons, activation=args.hidden_activation),
        Dense(args.num_classes, activation=args.output_activation)
    ])
    model.compile(optimizer=args.optimizer, loss=args.loss_function, metrics=args.metrics)
    return model

def main(args):
   args.results_path.mkdir(parents=True, exist_ok=True)

   (x_train, y_train, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, scaler) = preparing_data(
       data_path=None,
       feature_cols=args.feature_cols,
       input_window=args.input_window,
       leadtime=args.leadtime,
       scale=args.scale,
       verbose=args.verbose)
   
   model_datasplit = {
        "x_train":x_train,
        "y_train":y_train,
        "x_val":x_val,
        "y_val":y_val,
        "x_test":x_test,
        "y_test":y_test,
        "train_dates":train_dates,
        "val_dates":val_dates,
        "test_dates":test_dates,
        "scaler":scaler
    }
   
   model = build_model(model_datasplit.get("x_train").shape[1], model_datasplit.get("y_train").shape[1])
   early = EarlyStopping(monitor=args.monitor, patience=args.lr_reducer_patience, restore_best_weights=True)
   history = model.fit(
       model_datasplit.get("x_train"), model_datasplit.get("y_train"),
       validation_data=(model_datasplit.get("x_val"), model_datasplit.get("y_val")),
       epochs=args.epochs, batch_size=args.batch_size, callbacks=early, 
       verbose=args.verbose
   )
   print(f"trained {len(history.history['loss'])} epochs.")

def show_keys(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
