'''

inference_driver.py

This file will load in a keras file of a trained model and make predictions/inferences on a given dataset.
To change the path of the keras file, change the keras_path arg in configs.txt. 

Run from the repo root:
    python -m inference.driver.py
    
'''

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                             cohen_kappa_score, mean_absolute_error)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from src.data_prep.preparing_data import preparing_data

def inference(args):
    ''' Load the most recent keras file of a training run (the model and the scaler that it was trained with.) '''
    run_directory = sorted(args.keras_path.glob("run_*"))[-1]
    model = load_model(run_directory / "model.keras")
    scaler = joblib.load(run_directory / "scaler.joblib")

    (x_train, y_train, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, _) = preparing_data(
       data_path=None,
       feature_cols=args.feature_cols,
       input_window=args.input_window,
       leadtime=args.leadtime,
       scale=False,
       label_col="usdm_severity",   # ordinal target = severity rank (0 through 5)
       verbose=args.verbose)

    x_test = scaler.transform(x_test)   # apply the SAME scaler used during training

    # ordinal regression: the model outputs a continuous severity value, and rounds/clips it to a rank
    n_classes = len(args.class_names)
    # ravel flattens a multidimensional array into a 1D array
    raw = model.predict(x_test, verbose=0).ravel()

    y_pred = np.clip(np.rint(raw), 0, n_classes - 1).astype(int)
    y_true = y_test.argmax(axis=1)

    # metrics  
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mae = mean_absolute_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=range(n_classes),
                                  target_names=args.class_names,
                                  zero_division=0)
    print(f"test QWK={qwk:.3f}  macro-F1={macro_f1:.3f}  MAE={mae:.3f}  acc={acc:.3f}")

    (args.results_path / "metrics.txt").write_text(
       "ordinal regression on severity rank\n"
       f"QWK (primary): {qwk:.4f}\nmacro-F1: {macro_f1:.4f}\n"
       f"MAE (ranks): {mae:.4f}\naccuracy: {acc:.4f}\n\n{report}\n")

    pred_df = pd.DataFrame({
       "week_start": pd.to_datetime(test_dates),
       "true": [args.class_names[i] for i in y_true],
       "pred": [args.class_names[i] for i in y_pred],
       "true_index": y_true,
       "pred_index": y_pred,
       "raw_pred": raw.round(3),
        })
    
    pred_df.to_csv(args.results_path / "test_predictions.csv", index=False)