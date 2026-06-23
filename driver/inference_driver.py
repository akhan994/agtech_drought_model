# this driverr will take in keras files from trained models and will make predictions on an independent testing data  set

# testing

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from src.data_prep.preparing_data import preparing_data

def inference(args):

   # load the most recent training run (model + the scaler it was trained with)
   run_dir = sorted(args.keras_path.glob("run_*"))[-1]
   model = load_model(run_dir / "model.keras")
   scaler = joblib.load(run_dir / "scaler.joblib")

   (x_train, y_train, x_val, y_val, x_test, y_test, train_dates, val_dates, test_dates, _) = preparing_data(
       data_path=None,
       feature_cols=args.feature_cols,
       input_window=args.input_window,
       leadtime=args.leadtime,
       scale=False,
       verbose=args.verbose)

   x_test = scaler.transform(x_test)   # apply the SAME scaler used during training

   probs = model.predict(x_test, verbose=0)
   y_pred = probs.argmax(axis=1)
   y_true = y_test.argmax(axis=1)

   acc = accuracy_score(y_true, y_pred)
   macro_f1 = f1_score(y_true, y_pred, average="macro")
   report = classification_report(y_true, y_pred, labels=range(len(args.class_names)),
                                  target_names=args.class_names,
                                  zero_division=0)
   print(f"test accuracy={acc:.3f} macro-F1={macro_f1:.3f}")

   (args.results_path / "metrics.txt").write_text(
       f"test accuracy: {acc:.4f}\nmacro-F1: {macro_f1:.4f}\n\n{report}\n")
   
   pred_df = pd.DataFrame({
       "week_start": pd.to_datetime(test_dates),
       "true": [args.class_names[i] for i in y_true],
       "pred": [args.class_names[i] for i in y_pred],
       "true_idx": y_true,
       "pred_idx": y_pred,
   })
   
   for j, name in enumerate(args.class_names):
        pred_df[f"p_{name}"] = probs[:, j].round(4)

   pred_df.to_csv(args.results_path / "test_predictions.csv", index=False)
