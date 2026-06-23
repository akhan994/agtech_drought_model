"""
OPTION A: ordinal regression on the severity RANK (no_drought=0 ... D4=5).

Self-contained experiment to test "how much does ordering help?". It does NOT touch
the classification pipeline (preparing_data is used read-only), so reverting is just
deleting this file.

Idea: instead of a 6-way softmax classifier, predict a single continuous severity
value with MSE loss, then round to the nearest rank. Near-misses (D3 vs D4) cost
little; far-misses cost a lot -> the model is pushed to respect the ordering.

Run from the repo root:
    python -m scripts.run_ordinal_regression
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score, confusion_matrix, classification_report)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep.preparing_data import preparing_data

# severity order -- index == rank (this is the monotonic encoding, low -> high)
SEV_NAMES = ["no_drought", "D0", "D1", "D2", "D3", "D4"]
FEATURE_COLS = ["prcp", "tmax", "tmin", "ndvi", "pdsi", "phdi", "pmdi", "soil_moisture", "usdm_severity"]
INPUT_WINDOW = 4
LEADTIME = 4
EPOCHS = 200
OUT = Path("results") / ("ordinal_regression_" + datetime.now().strftime("%Y%m%d-%H%M%S"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Target = usdm_severity (the MONOTONIC rank). preparing_data one-hots it; we
    # argmax back to the integer rank 0..5 to use as a continuous regression target.
    (x_tr, y_tr, x_v, y_v, x_te, y_te,
     train_dates, val_dates, test_dates, scaler) = preparing_data(
        feature_cols=FEATURE_COLS, input_window=INPUT_WINDOW, leadtime=LEADTIME,
        scale=True, label_col="usdm_severity")
    y_tr = y_tr.argmax(axis=1).astype("float32")
    y_v = y_v.argmax(axis=1).astype("float32")
    y_te = y_te.argmax(axis=1).astype("int")

    # regression head: single linear output, MSE loss (vs 6-way softmax + crossentropy)
    model = Sequential([
        Input(shape=(x_tr.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(x_tr, y_tr, validation_data=(x_v, y_v),
              epochs=EPOCHS, batch_size=32, callbacks=[es], verbose=0)

    # decode: round to nearest rank, clip into the valid 0..5 range
    raw = model.predict(x_te, verbose=0).ravel()
    y_pred = np.clip(np.rint(raw), 0, 5).astype(int)

    # metrics: classification ones for comparison + ORDINAL ones that reward near-misses
    acc = accuracy_score(y_te, y_pred)
    macro_f1 = f1_score(y_te, y_pred, average="macro")
    mae = mean_absolute_error(y_te, y_pred)                      # avg #severity-levels off
    qwk = cohen_kappa_score(y_te, y_pred, weights="quadratic")   # ordinal agreement
    report = classification_report(y_te, y_pred, labels=range(6),
                                   target_names=SEV_NAMES, zero_division=0)
    print(f"acc={acc:.3f}  macroF1={macro_f1:.3f}  MAE_ranks={mae:.3f}  QWK={qwk:.3f}")

    (OUT / "metrics.txt").write_text(
        "OPTION A - ordinal regression on severity rank\n"
        f"accuracy: {acc:.4f}\nmacro-F1: {macro_f1:.4f}\n"
        f"MAE (ranks, avg levels off): {mae:.4f}\n"
        f"quadratic-weighted kappa: {qwk:.4f}\n\n{report}\n")

    pd.DataFrame({
        "week_start": pd.to_datetime(test_dates),
        "true_idx": y_te,
        "pred_idx": y_pred,
        "raw_pred": raw.round(3),
    }).to_csv(OUT / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_te, y_pred, labels=range(6))
    fig = px.imshow(cm, x=SEV_NAMES, y=SEV_NAMES, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="predicted", y="true", color="count"),
                    title=f"Ordinal regression (test)  acc={acc:.2f}  MAE={mae:.2f}  QWK={qwk:.2f}")
    fig.write_image(OUT / "confusion_matrix.png")
    fig.write_html(OUT / "confusion_matrix.html")

    print("saved ->", OUT)


if __name__ == "__main__":
    main()
