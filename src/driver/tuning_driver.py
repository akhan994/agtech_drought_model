"""
tuning_driver.py - hyperparameter tuning via RANDOM SEARCH (scikit-learn).

>>> PSEUDOCODE / DESIGN SPEC - not a finished implementation. <<<
(args/parser assumed fixed: this references args.* freely.)

GOAL
    Search over MLP hyperparameters to improve the drought classifier, then hand the
    best configuration to the training driver.

TWO PROJECT-SPECIFIC RULES THAT SHAPE THE DESIGN
    1. Score by MACRO-F1, not accuracy. The classes are imbalanced, so accuracy would
       reward a model that just predicts the common classes. Macro-F1 is the metric we
       actually care about (rare drought levels weighted equally).
    2. Respect TIME ORDER. We must NOT use a shuffled cross-validation, or the model
       sees the future -> leakage. Either evaluate on the existing chronological
       validation split, or use sklearn's TimeSeriesSplit (never plain KFold).

APPROACH (sklearn-native, minimal deps)
    sklearn.model_selection.ParameterSampler draws N random hyperparameter combos from a
    search space. For each combo we build + train the Keras model and score it on a
    time-ordered validation set. We rank the trials and save the best config.

    (Alternative: sklearn RandomizedSearchCV wrapping the model with
     scikeras.KerasClassifier. Cleaner API, but you MUST pass cv=TimeSeriesSplit(...)
     and scoring='f1_macro', and scikeras would need adding to environment.yml.
     Sketch at the bottom of this file.)
"""

# ---------------------------------------------------------------- imports (real)
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform                       # sample learning rates log-uniformly
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K

from src.data_prep.preparing_data import preparing_data
from src.driver.training_driver import build_model       # reuse - must be PARAMETRIZED
from src.helper.parser import create_parser


# ---------------------------------------------------------------- search space
def get_search_space():
    # dict of {hyperparameter: list of choices OR a scipy distribution to sample from}
    # ParameterSampler treats lists as discrete choices and distributions as continuous.
    return {
        "num_layers":    [1, 2, 3],
        "neurons":       [16, 32, 64, 128],
        "learning_rate": loguniform(1e-4, 1e-2),          # continuous, log scale
        "batch_size":    [16, 32, 64],
        "dropout":       [0.0, 0.2, 0.4],
        "input_window":  [4, 8, 12, 26],                  # NOTE: changes the data shape (see below)
    }


# ---------------------------------------------------------------- main routine
def tune(args):
    # 1. SETUP
    #    out_dir = Path(args.results_path) / "tuning";  out_dir.mkdir(parents=True, exist_ok=True)
    #    sampler = ParameterSampler(get_search_space(),
    #                               n_iter=args.n_iter,            # how many random combos to try
    #                               random_state=args.random_state)
    #
    #    OPTIMIZATION: if "input_window" is NOT in the search space, call preparing_data
    #    ONCE here (outside the loop) and reuse the arrays - far faster. It is only inside
    #    the loop below because window size changes the windowed data.

    # 2. LOOP OVER RANDOM TRIALS
    #    trials = []
    #    for trial_id, params in enumerate(sampler):
    #
    #        K.clear_session()                              # free memory/graph between trials
    #
    #        # (a) PREPARE DATA for this trial
    #        #     input_window is a hyperparameter, so re-window per trial:
    #        #     x_train, y_train, x_val, y_val, *_ = preparing_data(
    #        #         feature_cols=args.feature_cols,
    #        #         input_window=params["input_window"],
    #        #         leadtime=args.leadtime,
    #        #         scale=args.scale,
    #        #         num_classes=args.num_classes)
    #
    #        # (b) BUILD model from params (build_model must accept these knobs)
    #        #     model = build_model(input_dim=x_train.shape[1],
    #        #                         num_classes=args.num_classes,
    #        #                         num_layers=params["num_layers"],
    #        #                         neurons=params["neurons"],
    #        #                         dropout=params["dropout"],
    #        #                         learning_rate=params["learning_rate"])
    #
    #        # (c) TRAIN quietly, early-stop on val_loss
    #        #     history = model.fit(x_train, y_train,
    #        #                         validation_data=(x_val, y_val),
    #        #                         epochs=args.epochs,
    #        #                         batch_size=params["batch_size"],
    #        #                         callbacks=[EarlyStopping(...)],
    #        #                         class_weight=...,        # carry imbalance handling in here too
    #        #                         verbose=0)
    #
    #        # (d) SCORE on validation with MACRO-F1 (NOT accuracy)
    #        #     y_pred = model.predict(x_val, verbose=0).argmax(axis=1)
    #        #     y_true = y_val.argmax(axis=1)
    #        #     val_macro_f1 = f1_score(y_true, y_pred, average="macro")
    #
    #        # (e) RECORD the trial
    #        #     trials.append({"trial_id": trial_id, **params,
    #        #                    "val_macro_f1": val_macro_f1,
    #        #                    "epochs_run": len(history.history["loss"])})
    #        #     print(f"[trial {trial_id}] macroF1={val_macro_f1:.3f}  {params}")

    # 3. RANK & SAVE
    #    results = pd.DataFrame(trials).sort_values("val_macro_f1", ascending=False)
    #    results.to_csv(out_dir / "tuning_results.csv", index=False)
    #    best = results.iloc[0].to_dict()
    #    with open(out_dir / "best_hyperparameters.json", "w") as f:
    #        json.dump(best, f, indent=2)          # run_training can load this to train final model
    #    print("BEST:", best)

    # 4. (OPTIONAL) refit best config on train+val, evaluate ONCE on the held-out TEST set,
    #    and report final macro-F1 + confusion matrix. Keep test untouched until this step.
    pass


# ---------------------------------------------------------------- entry point
if __name__ == "__main__":
    args = create_parser().parse_args()       # parses @configs/configs.txt + CLI
    tune(args)


# =====================================================================================
# ALTERNATIVE SKETCH: sklearn RandomizedSearchCV + scikeras (if you prefer the sklearn API)
# -------------------------------------------------------------------------------------
# from scikeras.wrappers import KerasClassifier            # pip add: scikeras
# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
#
# clf = KerasClassifier(model=build_model, num_classes=args.num_classes, verbose=0)
# search = RandomizedSearchCV(
#     estimator=clf,
#     param_distributions=get_search_space(),
#     n_iter=args.n_iter,
#     scoring="f1_macro",                # <-- macro-F1, matches our objective
#     cv=TimeSeriesSplit(n_splits=5),    # <-- MUST be time-aware, NOT default KFold (leakage)
#     random_state=args.random_state,
# )
# search.fit(x_train, y_train)           # NOTE: input_window can't be tuned this way
# print(search.best_params_, search.best_score_)
# =====================================================================================
