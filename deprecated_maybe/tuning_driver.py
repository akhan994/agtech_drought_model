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
import datetime
from scipy.stats import loguniform                       # sample learning rates log-uniformly
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K

from src.data_prep.preparing_data import preparing_data
from src.driver.training_driver import build_model       # reuse - must be PARAMETRIZED
from src.helper.parser import create_parser


# ---------------------------------------------------------------- search space
def get_search_space(args):

    path_to_model_runs = f"{args.results_folder}/{args.model_type}_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # dict of {hyperparameter: list of choices OR a scipy distribution to sample from}
    # ParameterSampler treats lists as discrete choices and distributions as continuous.
    return {
        "num_layers":    args.layers_list,
        "neurons":       args.unit_list,
        "learning_rate": args.lr_rate,
        "dropout":       [0.0, 0.2, 0.4],
        "input_window":  args.input_window,
        # NOTE: changes the data shape (see below)
    }


# ---------------------------------------------------------------- main routine
def tune(args):

    out_dir = Path(f"{args.results_path}/tuning_" + datetime.now().strftime(%Y%m%d-%H%M%S,
                                                                            parents=True, exists_ok=True)

    sampler = ParameterSampler(get_search_space(), n_iter=args.n_iter, random_state=args.random_state)
   
    # loop over random trials
    trials = []

    for trial_id, params in enumerate(sampler):
        K.clear_session
        x_train, y_train, x_val, y_val, *_ = preparing_data(
            feature_cols=args.feature_cols,
            input_window=params["input_window"],
            leadtime=args.leadtime,
            scale=args.scale,
            num_classes=args.num_output_neurons
        )

        model = build_model(input_dim=x_train.shape[1],
                            num_classes=args.num_output_neurons,
                            num_layers=params["num_layers"],
                            neurons=params["neurons"],
                            dropout=params["dropout"],
                            learning_rate=params["learning_rate"]
                            )
        
        history = model.fit(x_train, 
                            y_train,
                            validation_data=(x_val, y_val),
                            epochs=args.epochs,
                            batch_size=params["batch_size"],
                            callbacks=args.call_back_monitor)
                            #class_weight=class_weight
        
        # score on validation with macro-f1
        y_pred = model.predict(x_val, verbose=0).argmax(axis=1)
        y_true = y_val.argmax(axis=1)
        val_macro_f1 = f1_score(y_true, y_pred, average="macro")

        # record trial
        trials.append({"trial_id":trial_id, 
                       **params, 
                       "val_macro_f1":val_macro_f1, 
                       "epochs_run":len(history.history["loss"])})
        
        print(f"[trial {trial_id}] macroF1={val_macro_f1:.3f} {params}")

        # rank and save 
        results = pd.DataFrame(trials).sort_values("val_macro_f1", ascending=False)
        results.to_csv(out_dir / "tuning_results.csv", index=False)
        best = results.iloc[0].to_dict()
        with open(out_dir / "best_hyperparameters.json", "w") as f:
            json.dump(best, f, indent=2)
            print("BEST:", best)

# ---------------------------------------------------------------- entry point
if __name__ == "__main__":
    args = create_parser().parse_args()       # parses @configs/configs.txt + CLI
    tune(args)
