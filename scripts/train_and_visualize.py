"""
Small end-to-end pipeline: prepare data -> train an MLP drought classifier ->
evaluate -> visualize. Deliberately minimal (a "tonight" baseline), separate from
the fuller config-driven training_driver.py.

Outputs (into results/baseline/):
  - metrics.txt              accuracy + macro-F1 + per-class report
  - loss_curve.png           train/val loss over epochs
  - confusion_matrix.png     test-set confusion matrix
  - test_predictions.csv     date, true class, predicted class, probabilities
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

CLASS_NAMES = ["D0", "D1", "D2", "D3", "D4", "no_drought"]   # index = class_number
OUT_DIR = Path("results/baseline")


def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    (x_train, y_train, x_val, y_val, x_test, y_test,
     train_dates, val_dates, test_dates, scaler) = preparing_data(
        data_path=None,
        feature_cols=["prcp", "tmax", "tmin", "ndvi"],
        input_window=4,
        leadtime=4,
        scale=True,
        verbose=1,
    )

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

    # ---- train ----
    model = build_model(x_train.shape[1], model_datasplit.get("y_train".shape[1])
    early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    history = model.fit(
        model_datasplit.get("x_train"), model_datasplit.get("y_train"),
        validation_data=(x_val, y_val),
        epochs=200, batch_size=32, callbacks=[early], verbose=0,
    )
    print(f"trained {len(history.history['loss'])} epochs")

    # ---- evaluate (test) ----
    probs = model.predict(x_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                   target_names=CLASS_NAMES, zero_division=0)
    print(f"test accuracy={acc:.3f}  macro-F1={macro_f1:.3f}")

    (OUT_DIR / "metrics.txt").write_text(
        f"test accuracy: {acc:.4f}\nmacro-F1: {macro_f1:.4f}\n\n{report}\n")

    # ---- predictions csv ----
    pred_df = pd.DataFrame({
        "week_start": pd.to_datetime(test_dates),
        "true": [CLASS_NAMES[i] for i in y_true],
        "pred": [CLASS_NAMES[i] for i in y_pred],
    })
    for j, name in enumerate(CLASS_NAMES):
        pred_df[f"p_{name}"] = probs[:, j].round(4)
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # ---- viz: loss curve ----
    loss_fig = go.Figure()
    loss_fig.add_scatter(y=history.history["loss"], name="train loss")
    loss_fig.add_scatter(y=history.history["val_loss"], name="val loss")
    loss_fig.update_layout(title="Training / validation loss",
                           xaxis_title="epoch", yaxis_title="categorical crossentropy")
    loss_fig.write_image(OUT_DIR / "loss_curve.png")
    loss_fig.write_html(OUT_DIR / "loss_curve.html")

    # ---- viz: confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_fig = px.imshow(cm, x=CLASS_NAMES, y=CLASS_NAMES, text_auto=True,
                       color_continuous_scale="Blues",
                       labels=dict(x="predicted", y="true", color="count"),
                       title=f"Confusion matrix (test)  acc={acc:.2f}  macroF1={macro_f1:.2f}")
    cm_fig.write_image(OUT_DIR / "confusion_matrix.png")
    cm_fig.write_html(OUT_DIR / "confusion_matrix.html")

    # ---- viz: drought severity over time (actual history + test forecast) ----
    # remap class_number to a severity rank so the axis reads low->high
    # (encoding has no_drought=5; here no_drought sits at the bottom)
    sev_order = ["no_drought", "D0", "D1", "D2", "D3", "D4"]   # least -> most severe
    def to_sev(cn):
        return 0 if cn == 5 else cn + 1

    hist = pd.read_csv(r"raw_data/USDM/USDM_labels.csv")
    hist["week_start"] = pd.to_datetime(hist["week_start"], format="%m/%d/%y")
    hist = hist.sort_values("week_start")

    ts = go.Figure()
    ts.add_scatter(x=hist["week_start"], y=hist["class_number"].map(to_sev), mode="lines",
                   name="actual (USDM, full history)", line=dict(color="lightgray"))
    ts.add_scatter(x=pred_df["week_start"], y=[to_sev(i) for i in y_true], mode="lines",
                   name="actual (test window)", line=dict(color="black"))
    ts.add_scatter(x=pred_df["week_start"], y=[to_sev(i) for i in y_pred], mode="markers",
                   name="predicted (test)", marker=dict(color="red", size=5))
    ts.update_yaxes(tickvals=list(range(6)), ticktext=sev_order)
    ts.update_layout(title="Drought severity over time: USDM actual vs. model 4-week forecast",
                     xaxis_title="week", yaxis_title="severity (low -> high)")
    ts.write_image(OUT_DIR / "severity_timeseries.png")
    ts.write_html(OUT_DIR / "severity_timeseries.html")

    print(f"wrote results -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
