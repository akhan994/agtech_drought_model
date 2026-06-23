"""
OPTION B - CORAL ordinal classifier.

Self-contained experiment (preparing_data used read-only). Reverting = delete this file.

CORAL idea: instead of 6-way softmax, predict K-1=5 cumulative thresholds
("is severity > k?") using a SHARED weight vector + one bias per threshold. The
shared weight keeps the thresholds rank-consistent (monotonic), which is the
advantage over plain Frank-Hall cumulative outputs. Decode = count thresholds
whose probability passes 0.5.

Compared to Option A (regression), this keeps a per-threshold probability structure
and usually reduces the "smear toward the middle" that pure regression introduces.

Run from the repo root:
    python -m scripts.run_ordinal_coral
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score, confusion_matrix, classification_report)

from src.data_prep.preparing_data import preparing_data

SEV_NAMES = ["no_drought", "D0", "D1", "D2", "D3", "D4"]   # index == rank
NUM_CLASSES = 6
FEATURE_COLS = ["prcp", "tmax", "tmin", "ndvi", "pdsi", "phdi", "pmdi", "soil_moisture", "usdm_severity"]
INPUT_WINDOW = 4
LEADTIME = 4
EPOCHS = 200
OUT = Path("results") / ("ordinal_coral_" + datetime.now().strftime("%Y%m%d-%H%M%S"))


class CoralOutput(layers.Layer):
    """K-1 cumulative-threshold logits that SHARE one weight vector (CORAL).

    logit_k = w . x + b_k   -- only the bias differs per threshold, which keeps the
    predicted P(y>0) >= P(y>1) >= ... rank-consistent.
    """
    def __init__(self, num_classes, **kw):
        super().__init__(**kw)
        self.num_thresholds = num_classes - 1

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True, name="coral_w")
        self.b = self.add_weight(shape=(self.num_thresholds,),
                                 initializer="zeros", trainable=True, name="coral_b")

    def call(self, x):
        return tf.matmul(x, self.w) + self.b      # (batch, K-1) logits


def to_levels(ranks, num_classes):
    """rank r -> cumulative binary [1 if r>k else 0 for k in 0..K-2]."""
    ks = np.arange(num_classes - 1)
    return (ranks[:, None] > ks).astype("float32")


def decode(logits):
    """predicted rank = number of thresholds whose sigmoid passes 0.5."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs > 0.5).sum(axis=1).astype(int)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # target = usdm_severity (monotonic rank). preparing_data one-hots it; argmax -> rank.
    (x_tr, y_tr, x_v, y_v, x_te, y_te,
     train_dates, val_dates, test_dates, scaler) = preparing_data(
        feature_cols=FEATURE_COLS, input_window=INPUT_WINDOW, leadtime=LEADTIME,
        scale=True, label_col="usdm_severity")
    r_tr, r_v, r_te = y_tr.argmax(1), y_v.argmax(1), y_te.argmax(1).astype(int)

    L_tr = to_levels(r_tr, NUM_CLASSES)   # cumulative binary targets
    L_v = to_levels(r_v, NUM_CLASSES)

    model = Sequential([
        layers.Input(shape=(x_tr.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        CoralOutput(NUM_CLASSES),
    ])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(x_tr, L_tr, validation_data=(x_v, L_v),
              epochs=EPOCHS, batch_size=32, callbacks=[es], verbose=0)

    y_pred = decode(model.predict(x_te, verbose=0))

    acc = accuracy_score(r_te, y_pred)
    macro_f1 = f1_score(r_te, y_pred, average="macro")
    mae = mean_absolute_error(r_te, y_pred)
    qwk = cohen_kappa_score(r_te, y_pred, weights="quadratic")
    report = classification_report(r_te, y_pred, labels=range(6),
                                   target_names=SEV_NAMES, zero_division=0)
    print(f"acc={acc:.3f}  macroF1={macro_f1:.3f}  MAE_ranks={mae:.3f}  QWK={qwk:.3f}")

    (OUT / "metrics.txt").write_text(
        "OPTION B - CORAL ordinal classifier\n"
        f"accuracy: {acc:.4f}\nmacro-F1: {macro_f1:.4f}\n"
        f"MAE (ranks, avg levels off): {mae:.4f}\n"
        f"quadratic-weighted kappa: {qwk:.4f}\n\n{report}\n")

    pd.DataFrame({
        "week_start": pd.to_datetime(test_dates),
        "true_idx": r_te,
        "pred_idx": y_pred,
    }).to_csv(OUT / "test_predictions.csv", index=False)

    cm = confusion_matrix(r_te, y_pred, labels=range(6))
    fig = px.imshow(cm, x=SEV_NAMES, y=SEV_NAMES, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="predicted", y="true", color="count"),
                    title=f"CORAL ordinal (test)  acc={acc:.2f}  MAE={mae:.2f}  QWK={qwk:.2f}")
    fig.write_image(OUT / "confusion_matrix.png")
    fig.write_html(OUT / "confusion_matrix.html")

    print("saved ->", OUT)


if __name__ == "__main__":
    main()
