# this driver will train the models 

# analogous to operational_mse_crps_driver.py from uq4ml_watertemp'

"""

The purpose of this script is to train a drought classification model. 

"""

from pathlib import Path
import joblib
import numpy as np 
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep.preparing_data import preparing_data

def build_model(input_dim, args):
    # ordinal regression: a single LINEAR output predicting the severity RANK,
    # trained with MSE (we solve the ordinal-classification task with a regression head).
    model = Sequential([Input(shape=(input_dim,))])
    for _ in range(args.num_layers):
        model.add(Dense(args.neurons, activation=args.hidden_activation))
        if args.dropout:
            model.add(Dropout(args.dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=Adam(learning_rate=args.lrate), loss=args.loss_function, metrics=args.metrics)
    return model

def main(args):
   args.results_path.mkdir(parents=True, exist_ok=True)
   args.keras_path.mkdir(parents=True, exist_ok=True)

   (x_train, y_train, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, scaler) = preparing_data(
       data_path=None,
       feature_cols=args.feature_cols,
       input_window=args.input_window,
       leadtime=args.leadtime,
       scale=args.scale,
       label_col="usdm_severity",   # ordinal target = monotonic severity rank (0..5)
       verbose=args.verbose)

   # regression target = the severity rank as a float (recover it from the one-hot)
   y_train = y_train.argmax(axis=1).astype("float32")
   y_val = y_val.argmax(axis=1).astype("float32")

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
   
   model = build_model(model_datasplit.get("x_train").shape[1], args)

   early = EarlyStopping(monitor=args.call_back_monitor, patience=args.early_stop_patience, restore_best_weights=True)
   history = model.fit(
       model_datasplit.get("x_train"), model_datasplit.get("y_train"),
       validation_data=(model_datasplit.get("x_val"), model_datasplit.get("y_val")),
       epochs=args.epochs, batch_size=args.batch_size, callbacks=[early], 
       verbose=args.verbose
   )
   
   # per-run directory: the timestamp keeps runs distinct, while the fixed
   # filenames inside stay findable by the inference / visualization drivers.
   run_dir = args.keras_path / f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
   run_dir.mkdir(parents=True, exist_ok=True)
   model.save(run_dir / "model.keras")
   joblib.dump(scaler, run_dir / "scaler.joblib")

   history_df = pd.DataFrame(history.history)
   history_df.to_csv(args.results_path/"history.csv", index=False)

   print(f"trained {len(history.history['loss'])} epochs.")

def show_keys(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
