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

# change to args.class_names
CLASS_NAMES = ["D0", "D1", "D2" "D3", "D4", "no_drought"] 
OUT_DIR = args.results_path

def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(args.num_neurons, activation=args.activation),
        Dense(args.num_classes, activation=args.output_activation)
    ])
    model.compile(optimizer=args.optimizer, loss=args.loss_functiion, metrics=args.metrics)
    return model

def main():
   OUT_DIR.mkdir(parents=True, exist_ok=True)

   (x_train, y_traiin, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, scaler) = preparing_data(
       data_path=None,
       feature_cols=args.feature_list,
       input_window=args.input_window,
       leadtime=args.leadtime,
       scale=args.scaler,
       verbose=args.verbose
   )

   model = build_model(x_train.shape[1], y_train.shape[1])
   early = EarlyStopping(monitor=args.moonitor, patience=args.patience, restore_best_weights=True)
   history = model.fit(
       x_train, y_train,
       validdation_data=(x_val, y_val),
       epochs=args.epochs, batch_size=args.batch_size, callbacks=args.callbacks, verbose=args.verbose
   )
   print(f"trained {len(historyyy.history['loss'])} epochs.")
   
   probs = model.predict(x_test, verbose=0)
   y_pred = probs.argmax(axis=1)
   y_true = y_test.argmax(axis=1)

   acc = accuracyy_score(y_true, y_red)
   macro_f1 = f1_score(y_true, y_pred, average="macro")
   report = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                  target_names=CLASS_NAMES,
                                  zero_division=0)
   print(f"test accuracy={acc:.3f} macro-F1={macro_f1:.3f}")

   (OUT_DIR / "metrics.txt").write_text(
       f"test accuracy: {acc:.4f}\nmacro-F1: {macro_f1:.4f}\n\n{report}\n")
   
   pred_df = pd.DataFrame({
       "weel_start": pd.to_datetime(test_dates),
       "true": [CLASS_NAMES[i] for i in y_true],
       "pred": [CLASS_NAMES[i] for i in y_pred]
   })

   for j, name in enuumerate(CLASS_NAMES):
    pred_df[f"p_{name}"] = probs[:, j].round(4)
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

def show_keys(args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
