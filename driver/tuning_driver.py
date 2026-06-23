"""
tuning_driver_randomsearch.py  -  READY-TO-RUN hyperparameter tuning.

Structure (the nested design you described):
    OUTER grid over input_window      -> changes the DATA SHAPE, so it must be a
                                         manual loop (RandomizedSearchCV can't tune it).
    INNER RandomizedSearchCV          -> tunes the MODEL hyperparameters, with:
        - scikeras KerasClassifier wrapping the MLP (makes it sklearn-compatible)
        - StandardScaler inside a Pipeline -> scaler is fit PER CV FOLD (no leakage)
        - cv = TimeSeriesSplit          -> chronological folds (no future leakage)
        - scoring = "f1_macro"          -> the metric we care about (class imbalance)

Run from the repo root:
    python -m driver.tuning_driver_randomsearch

Requires `scikeras` (added to environment.yml):
    conda run -n drought pip install scikeras

NOTE: this is a SEPARATE file from your manual `tuning_driver.py`; it's the
RandomizedSearchCV alternative so you can compare. Settings are constants below
(wire them to the parser/config later).
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
from scikeras.wrappers import KerasClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.data_prep.preparing_data import preparing_data

# ----------------------------- settings (edit here) --------------------------
FEATURE_COLS = ["prcp", "tmax", "tmin", "ndvi", "usdm_severity"]
LEADTIME     = 4
NUM_CLASSES  = 6
WINDOW_SIZES = [4, 8, 12]      # OUTER grid (data-shape hyperparameter)
N_ITER       = 10              # random model-combos tried per window
CV_SPLITS    = 3              # TimeSeriesSplit folds
EPOCHS       = 40
RANDOM_STATE = 0
RESULTS_DIR  = Path("results") / ("tuning_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

# inner search space. Pipeline step is named "clf", so keys are "clf__<param>";
# scikeras routes num_layers/neurons/dropout/learning_rate into build_model().
PARAM_SPACE = {
    "clf__num_layers":    [1, 2, 3],
    "clf__neurons":       [16, 32, 64, 128],
    "clf__dropout":       [0.0, 0.2, 0.4],
    "clf__learning_rate": loguniform(1e-4, 1e-2),
    "clf__batch_size":    [16, 32, 64],
}


# ----------------------------- model builder (scikeras-compatible) -----------
def build_model(meta, num_layers, neurons, dropout, learning_rate):
    # scikeras injects `meta` automatically; it carries the fitted data's shape.
    n_features = meta["n_features_in_"]
    n_classes = meta["n_classes_"]
    model = Sequential([Input(shape=(n_features,))])
    for _ in range(num_layers):
        model.add(Dense(neurons, activation="relu"))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation="softmax"))
    # sparse_* lets us feed INTEGER labels (no one-hot); equivalent to
    # categorical_crossentropy for single-label classification.
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------- main ------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows, best_overall = [], None

    for W in WINDOW_SIZES:
        # build the windowed data for THIS window size; scale=False because the
        # Pipeline's StandardScaler handles scaling inside each CV fold.
        x_tr, y_tr, x_va, y_va, x_te, y_te, *_ = preparing_data(
            feature_cols=FEATURE_COLS, input_window=W, leadtime=LEADTIME,
            scale=False, num_classes=NUM_CLASSES)

        # CV pool = train + val (still chronological: train precedes val).
        # test is left untouched for a later final evaluation.
        X = np.concatenate([x_tr, x_va])
        y = np.concatenate([y_tr, y_va]).argmax(axis=1)   # integer labels for scikeras

        clf = KerasClassifier(
            model=build_model, epochs=EPOCHS, verbose=0,
            class_weight="balanced",   # inverse-frequency weights, computed PER FOLD -> counters imbalance
            num_layers=1, neurons=32, dropout=0.0, learning_rate=1e-3,  # defaults (tuned below)
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        search = RandomizedSearchCV(pipe, 
                                    PARAM_SPACE, 
                                    n_iter=N_ITER, 
                                    scoring="f1_macro",
                                    cv=TimeSeriesSplit(n_splits=CV_SPLITS), 
                                    random_state=RANDOM_STATE,
                                    n_jobs=1, 
                                    refit=False, 
                                    verbose=1,   # n_jobs=1: TF isn't fork-safe
        )
        search.fit(X, y)

        print(f"[window {W}] best macro-F1={search.best_score_:.3f}  {search.best_params_}")

        for params, score in zip(search.cv_results_["params"],
                                 search.cv_results_["mean_test_score"]):
            all_rows.append({"input_window": W, "mean_macro_f1": score, **params})

        cand = {"input_window": W, "mean_macro_f1": float(search.best_score_),
                **search.best_params_}
        if best_overall is None or cand["mean_macro_f1"] > best_overall["mean_macro_f1"]:
            best_overall = cand

    # rank every (window, combo) trial and save
    results = pd.DataFrame(all_rows).sort_values("mean_macro_f1", ascending=False)
    results.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)
    with open(RESULTS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_overall, f, indent=2, default=str)

    print("\nBEST OVERALL:", best_overall)
    print("saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
