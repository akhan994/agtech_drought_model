"""

This tuning driver scores each combo by Quadratic-Weighted Kappa (QWK).
QWK directly rewards staying close on the severity scale, which is the point of going ordinal.

Because the regressor outputs continuous values, the scorer ROUNDS to the nearest
rank before computing kappa (you cannot pass raw floats to cohen_kappa_score).

Run from the repo root:
    python -m driver.tuning_driver
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, cohen_kappa_score, f1_score
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.data_prep.preparing_data import preparing_data

# ----------------------------- settings (edit here) --------------------------
FEATURE_COLS = ["prcp", "tmax", "tmin", "ndvi", "pdsi", "phdi", "pmdi", "soil_moisture", "usdm_severity"]
LEADTIME     = 4
NUM_CLASSES  = 6
WINDOW_SIZES = [4, 8, 12, 24]   # OUTER grid (input_window)
N_ITER       = 30               # random model-combos tried per window
CV_SPLITS    = 3                # TimeSeriesSplit folds
EPOCHS       = 200              # cap only -- EarlyStopping usually stops far sooner
EARLY_STOP_PATIENCE = 15
RANDOM_STATE = 0
RESULTS_DIR  = Path("results") / ("tuning_ordinal_qwk_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

# inner search space. Pipeline step is "reg", so keys are "reg__<param>".
PARAM_SPACE = {
    "reg__num_layers":    [1, 2, 3],
    "reg__neurons":       [16, 32, 64, 128],
    "reg__activation":    ["relu", "selu", "leaky_relu"],
    "reg__dropout":       [0.0, 0.2, 0.4],
    "reg__learning_rate": loguniform(1e-4, 1e-2),
    "reg__batch_size":    [16, 32, 64],
}


# ----------------------------- model builder (regression head) ---------------
def build_model(meta, num_layers, neurons, activation, dropout, learning_rate):
    n_features = meta["n_features_in_"]
    model = Sequential([Input(shape=(n_features,))])
    for _ in range(num_layers):
        model.add(Dense(neurons, activation=activation))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))     # single continuous severity output
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


# ----------------------------- ordinal QWK scorer ----------------------------
def _round_ranks(y_true, y_pred):
    yp = np.clip(np.rint(np.ravel(y_pred)), 0, NUM_CLASSES - 1).astype(int)
    yt = np.rint(np.ravel(y_true)).astype(int)
    return yt, yp

def ordinal_qwk(y_true, y_pred):
    """Round continuous predictions to ranks, then QWK."""
    yt, yp = _round_ranks(y_true, y_pred)
    return cohen_kappa_score(yt, yp, weights="quadratic")

def ordinal_macro_f1(y_true, y_pred):
    """Round continuous predictions to ranks, then macro-F1."""
    yt, yp = _round_ranks(y_true, y_pred)
    return f1_score(yt, yp, average="macro")

# QWK is the SELECTION metric (combos are ranked/picked by it). macro_f1 is tracked
# alongside for human analysis only -- it does NOT influence which combo is chosen.
SCORERS = {"qwk": make_scorer(ordinal_qwk), "macro_f1": make_scorer(ordinal_macro_f1)}


# ----------------------------- main ------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, best_overall = [], None

    for W in WINDOW_SIZES:
        x_tr, y_tr, x_va, y_va, x_te, y_te, *_ = preparing_data(
            feature_cols=FEATURE_COLS, input_window=W, leadtime=LEADTIME,
            scale=False, label_col="usdm_severity", num_classes=NUM_CLASSES)

        X = np.concatenate([x_tr, x_va])
        # target = the severity rank as a float (recover from one-hot via argmax)
        y = np.concatenate([y_tr, y_va]).argmax(axis=1).astype("float32")

        reg = KerasRegressor(
            model=build_model, epochs=EPOCHS, verbose=0,
            validation_split=0.15,
            callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                     restore_best_weights=True)],
            num_layers=1, neurons=32, activation="relu", dropout=0.0, learning_rate=1e-3,
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])

        # multi-metric: rank by QWK, but record macro-F1 for every combo too.
        # (refit=False + multi-metric -> best_* attrs aren't set, so we pick the
        #  winner ourselves via argmax of the QWK column in cv_results_.)
        search = RandomizedSearchCV(pipe, PARAM_SPACE, n_iter=N_ITER, scoring=SCORERS,
                                    refit=False, cv=TimeSeriesSplit(n_splits=CV_SPLITS),
                                    random_state=RANDOM_STATE, n_jobs=1, verbose=1)
        search.fit(X, y)

        cvr = search.cv_results_
        qwk_scores, mf1_scores = cvr["mean_test_qwk"], cvr["mean_test_macro_f1"]
        best_i = int(np.argmax(qwk_scores))
        print(f"[window {W}] best QWK={qwk_scores[best_i]:.3f} "
              f"(macroF1={mf1_scores[best_i]:.3f})  {cvr['params'][best_i]}")

        for params, q, m in zip(cvr["params"], qwk_scores, mf1_scores):
            all_rows.append({"input_window": W, "mean_qwk": q, "mean_macro_f1": m, **params})

        cand = {"input_window": W, "mean_qwk": float(qwk_scores[best_i]),
                "mean_macro_f1": float(mf1_scores[best_i]), **cvr["params"][best_i]}
        if best_overall is None or cand["mean_qwk"] > best_overall["mean_qwk"]:
            best_overall = cand

    results = pd.DataFrame(all_rows).sort_values("mean_qwk", ascending=False)
    results.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)
    with open(RESULTS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_overall, f, indent=2, default=str)

    print("\nBEST OVERALL:", best_overall)
    print("saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
