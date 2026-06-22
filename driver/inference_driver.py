# this driverr will take in keras files from trained models and will make predictions on an independent testing data  set

# testing

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

def inference(model, model_datasplit, args):
   probs = model.predict(model_datasplit.get("x_test"), verbose=0)
   y_pred = probs.argmax(axis=1)
   y_true = model_datasplit.get("y_test").argmax(axis=1)

   acc = accuracy_score(y_true, y_pred)
   macro_f1 = f1_score(y_true, y_pred, average="macro")
   report = classification_report(y_true, y_pred, labels=range(len(args.class_names)),
                                  target_names=args.class_names,
                                  zero_division=0)
   print(f"test accuracy={acc:.3f} macro-F1={macro_f1:.3f}")

   (args.results_path / "metrics.txt").write_text(
       f"test accuracy: {acc:.4f}\nmacro-F1: {macro_f1:.4f}\n\n{report}\n")
   
   pred_df = pd.DataFrame({
       "week_start": pd.to_datetime(model_datasplit.get("test_dates")),
       "true": [args.class_names[i] for i in y_true],
       "pred": [args.class_names[i] for i in y_pred]
   })

   for j, name in enumerate(args.class_names):
    pred_df[f"p_{name}"] = probs[:, j].round(4)
    pred_df.to_csv(args.results_path / "test_predictions.csv", index=False)

    