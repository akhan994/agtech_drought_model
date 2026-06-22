"""
explainability_driver.py - post-hoc XAI for the trained drought classifier.

>>> PSEUDOCODE / DESIGN SPEC - not a finished implementation. <<<
(args/parser assumed fixed; references args.* freely.)

WHAT THIS DOES
    Loads an ALREADY-TRAINED model + scaler (like inference_driver) and explains it.
    NONE of these methods retrain the model - they only call model.predict.
        #1  Permutation feature importance   (global)
        #2  SHAP values                       (global + local)
        #4  Partial dependence                (how one feature drives predictions)

TWO RULES THAT SHAPE EVERYTHING
    1. Use the SAME scaler the model was trained with. The model only understands
       scaled inputs, so explanations live in the scaled feature space.
    2. Score / explain on the TEST (or VAL) set, with MACRO-F1 as the metric
       (imbalanced classes - accuracy would mislead).

See docs/xai_notes.txt for the concepts behind each method.
"""

# ---------------------------------------------------------------- imports (real)
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
# import shap                                  # for #2 (add `shap` to environment.yml)

from src.data_prep.preparing_data import preparing_data

CLASS_NAMES = ["D0", "D1", "D2", "D3", "D4", "no_drought"]   # index = class_number


# ---------------------------------------------------------------- helpers
def feature_names(feature_cols, input_window):
    # X is a flattened (W, F) window: flatten() lays it out row-major, so
    #   column k  ->  week (k // F), feature (k % F).
    # Build readable labels like "prcp_w-4 ... prcp_w-1" so plots make sense.
    # W = input_window (most recent week is w-1), F = len(feature_cols).
    # returns a list of length W*F, e.g. ["prcp_w-4","tmax_w-4",...,"ndvi_w-1"]
    ...


def macro_f1(model, X, y_onehot):
    # y_pred = model.predict(X, verbose=0).argmax(axis=1)
    # y_true = y_onehot.argmax(axis=1)
    # return f1_score(y_true, y_pred, average="macro")
    ...


def load_artifacts(args):
    # model  = load_model(args.model_path)          # the saved .keras file
    # scaler = joblib.load(args.scaler_path)        # the StandardScaler from training
    # return model, scaler
    ...


def get_test_data(args):
    # Reuse preparing_data with the SAME settings the model was trained on
    # (feature_cols, input_window, leadtime, num_classes). Pass scale=False here and
    # apply the LOADED scaler yourself, so train/explain use identical scaling.
    #   out = preparing_data(feature_cols=args.feature_cols, input_window=args.input_window,
    #                        leadtime=args.leadtime, scale=False, num_classes=args.num_classes)
    #   x_train, y_train, x_val, y_val, x_test, y_test, *_ = out
    #   return x_train, x_test, y_test           # x_train sample is the SHAP background
    ...


# ---------------------------------------------------------------- #1 permutation importance
def permutation_importance_manual(model, x_test, y_test, names, n_repeats=10, groups=None):
    # baseline = macro_f1(model, x_test, y_test)
    # importances = {}
    # for each feature column j (or each GROUP of columns in `groups`):
    #     drops = []
    #     for _ in range(n_repeats):
    #         x_perm = x_test.copy()
    #         shuffle x_perm[:, j] across the ROW axis (np.random.permutation of that column)
    #         (for a group: shuffle all its columns together with the SAME row order)
    #         drops.append(baseline - macro_f1(model, x_perm, y_test))
    #     importances[name_j] = mean(drops)
    # -> bar chart (px.bar) sorted desc; save to results/explainability/perm_importance.png
    #
    # WHY MANUAL: sklearn.inspection.permutation_importance needs a sklearn estimator;
    # a raw Keras model isn't one. This loop is the same idea, dependency-free.
    # TIP: pass `groups` = {"prcp":[cols...], "tmax":[...]} to get per-VARIABLE importance
    #      instead of noisy per-lag numbers (our lags are correlated).
    ...


# ---------------------------------------------------------------- #2 SHAP
def shap_explain(model, x_train, x_test, names, class_index, k=100):
    # background = shap.sample(x_train, 100)             # reference distribution
    # explainer  = shap.KernelExplainer(model.predict, background)   # model-agnostic
    #   (or shap.DeepExplainer(model, background) - faster for NNs)
    # shap_values = explainer.shap_values(x_test[:k])    # explain k test rows
    #
    # MULTICLASS: shap_values is a LIST (one array per class). Pick class_index
    # (e.g. CLASS_NAMES.index("D4")) for the plots below.
    #   shap.summary_plot(shap_values[class_index], x_test[:k], feature_names=names)  # GLOBAL
    #   shap.waterfall_plot(... single row ...)                                       # LOCAL
    # save figures to results/explainability/
    #
    # NOTE: needs `shap` installed (add to environment.yml).
    ...


# ---------------------------------------------------------------- #4 partial dependence
def partial_dependence_manual(model, x_test, names, feature_idx, class_index, n_points=20):
    # For ONE feature column `feature_idx` and ONE class `class_index`:
    #   grid = np.linspace(x_test[:, feature_idx].min(), .max(), n_points)
    #   avg_prob = []
    #   for v in grid:
    #       x_tmp = x_test.copy(); x_tmp[:, feature_idx] = v
    #       probs = model.predict(x_tmp, verbose=0)[:, class_index]
    #       avg_prob.append(probs.mean())
    #   -> line plot avg_prob vs grid; save to results/explainability/pdp_<name>.png
    #
    # WHY MANUAL: sklearn's PartialDependenceDisplay needs a sklearn estimator; this
    # loop is the PDP definition and works directly on the Keras model.
    # NOTE: grid is in SCALED units (model space). For a readable x-axis, inverse-transform
    # via the scaler, or label the axis "scaled <feature>".
    ...


# ---------------------------------------------------------------- main
def explain(args):
    # out_dir = Path(args.results_path) / "explainability"; out_dir.mkdir(parents=True, exist_ok=True)
    # model, scaler = load_artifacts(args)
    # x_train, x_test, y_test = get_test_data(args)
    # x_train = scaler.transform(x_train); x_test = scaler.transform(x_test)   # SAME scaler
    # names = feature_names(args.feature_cols, args.input_window)
    #
    # permutation_importance_manual(model, x_test, y_test, names, groups=...)        # #1
    # shap_explain(model, x_train, x_test, names, class_index=CLASS_NAMES.index("D4")) # #2
    # for j in top_features: partial_dependence_manual(model, x_test, names, j, class_index=...)  # #4
    pass


if __name__ == "__main__":
    from src.helper.parser import create_parser
    args = create_parser().parse_args()    # needs --model_path / --scaler_path args
    explain(args)
