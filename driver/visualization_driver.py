import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep.preparing_data import preparing_data

def loss_curve(args):

    history = pd.read_csv(args.results_path/"history.csv")

    loss_fig = go.Figure()
    loss_fig.add_scatter(y=history["loss"], name="train loss")
    loss_fig.add_scatter(y=history["val_loss"], name="val loss")
    loss_fig.update_layout(title="Training / validation loss",
                           xaxis_title="epoch", yaxis_title="MSE")
    loss_fig.write_image(args.results_path / "loss_curve.png")
    loss_fig.write_html(args.results_path / "loss_curve.html")

def plot_confusion_matrix(args):

    df = pd.read_csv(args.results_path/"test_predictions.csv")

    y_true = df["true_idx"]
    y_pred = df["pred_idx"]

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(args.class_names)))
    cm_fig = px.imshow(cm, x=args.class_names, y=args.class_names, text_auto=True,
                       color_continuous_scale="Blues",
                       labels=dict(x="predicted", y="true", color="count"),
                       title=f"Confusion matrix (test)  QWK={qwk:.2f}  macroF1={macro_f1:.2f}")
    cm_fig.write_image(args.results_path / "confusion_matrix.png")
    cm_fig.write_html(args.results_path / "confusion_matrix.html")

def time_series(args):
    ''' ---- viz: drought severity over time (actual history + test forecast) ----
    remap class_number to a severity rank so the axis reads low->high
    (encoding has no_drought=5; here no_drought sits at the bottom) '''

    df = pd.read_csv(args.results_path/"test_predictions.csv")
    y_true = df["true_idx"]    # already a severity rank (0=no_drought .. 5=D4)
    y_pred = df["pred_idx"]

    sev_order = ["no_drought", "D0", "D1", "D2", "D3", "D4"]   # least -> most severe
    # the USDM history file uses the OTHER encoding (no_drought=5), so remap it to a
    # severity rank for the gray line; the test true/pred are already severity ranks.
    def to_sev(cn):
        return 0 if cn == 5 else cn + 1

    hist = pd.read_csv(r"raw_data/USDM/USDM_labels.csv")
    hist["week_start"] = pd.to_datetime(hist["week_start"], format="%m/%d/%y")
    hist = hist.sort_values("week_start")

    ts = go.Figure()
    ts.add_scatter(x=hist["week_start"], y=hist["class_number"].map(to_sev), mode="lines",
                   name="actual (USDM, full history)", line=dict(color="lightgray"))
    ts.add_scatter(x=df["week_start"], y=y_true, mode="lines",
                   name="actual (test window)", line=dict(color="black"))
    ts.add_scatter(x=df["week_start"], y=y_pred, mode="markers",
                   name="predicted (test)", marker=dict(color="red", size=5))
    ts.update_yaxes(tickvals=list(range(6)), ticktext=sev_order)
    ts.update_layout(title="Drought severity over time: USDM actual vs. model 4-week forecast",
                     xaxis_title="week", yaxis_title="severity (low -> high)")
    ts.write_image(args.results_path / "severity_timeseries.png")
    ts.write_html(args.results_path / "severity_timeseries.html")

    print(f"wrote results -> {args.results_path}/")
