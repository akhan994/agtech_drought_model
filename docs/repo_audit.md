# Repo Audit — drought model (updated 2026-06-22)

_Re-verified against the actual code (not just the DONE marks). Line numbers drift as you edit._

---

## ✅ Completed since last audit

- ~~`parser.py` never returns the parser~~ → **fixed** (`return parser`).
- ~~`results_path` is a str but used as a Path~~ → **fixed** (`type=Path`).
- ~~`--metrics` default `[0]`~~ → **fixed** (now `None`).
- ~~`--feature_cols` / `--class_names` missing from parser~~ → **added**.
- ~~`--activation_function` vs config mismatch~~ → **renamed to `--hidden_activation`** (parser + config + driver agree).
- ~~`--num_classes` vs `--num_output_neurons` drift~~ → **unified on `num_output_neurons`**.
- ~~`config --scale=true` / `--v` / `--class_models`~~ → **fixed** (`--scale` bare, `-v`, `--class_names`).
- ~~`training_logger.py` missing `import tf`~~ → **fixed**.
- ~~`callbacks=early` not a list~~ → **fixed** (`callbacks=[early]`).
- ~~EarlyStopping used `args.monitor` / wrong patience~~ → **fixed** (`call_back_monitor` + `early_stop_patience`).
- ~~`build_model` called without `args`~~ → **args now passed** (but see A1 — input_dim is now wrong).
- ~~`inference_driver` `to_csv` inside the loop~~ → **fixed** (de-indented).
- ~~`visualization_driver.loss_curve` undefined names + dup figure~~ → **fixed** (and split into 3 functions).
- ~~Two driver locations / stub `tuning_driver`~~ → **consolidated into root `driver/`**.
- **New:** `class_weight="balanced"` added to `tuning_driver_randomsearch.py`; it runs end-to-end.

---

## A. Still-blocking bugs

1. **`training_driver` passes a TUPLE as `input_dim`.** Line ~58:
   `build_model((x_train.shape[1], y_train.shape[1]), args)` → inside, `Input(shape=(input_dim,))`
   becomes `Input(shape=((20,6),))`. Pass just the feature count: `build_model(x_train.shape[1], args)`.
   (The `args` half is fixed; the input_dim half got broken.)

2. **`model.save(...)` line has 3 problems** (training_driver ~line 68):
   - `args.keras_path` — **parser defines `--keras_files_path`, not `keras_path`** → `AttributeError`.
   - `datetime.now()` — file does `import datetime`, so it must be `datetime.datetime.now()` (or
     `from datetime import datetime`). Same bug sits in `tuning_driver.py`.
   - the keras output dir is never created (`.mkdir`), so the save path may not exist.

3. **`visualization_driver.confusion_matrix()` is doubly broken:**
   - the function name **shadows the imported `confusion_matrix`** from sklearn (line 7), so the call
     on line ~24 recurses into itself instead of sklearn. Rename the function (e.g. `plot_confusion`).
   - references `acc` and `macro_f1` (line ~28) that are **not passed in** → `NameError`.

4. **`visualization_driver.time_series()` references `y_pred`** (line ~49) which isn't a parameter
   (only `pred_df, y_true, args`) → `NameError`. Pass `y_pred` in.

5. **Manual `tuning_driver.py` is still broken / unchanged** (you've been using the randomsearch one):
   - **syntax error** at ~line 66 (unquoted `strftime` format, mismatched parens, `mkdir` args jammed
     into `Path(...)`) — the file won't even import.
   - `import datetime` → `datetime.now()` fails.
   - `from src.driver.training_driver import build_model` — wrong path (drivers are in root `driver/`).
   - `get_search_space()` called without its `args` argument (line ~69).
   - `K.clear_session` missing `()`.
   - `build_model(input_dim=..., num_layers=..., ...)` — but `build_model` is `(input_dim, args)`,
     it doesn't take those kwargs (the "parametrize build_model" prerequisite isn't done).
   - `params["batch_size"]` — `batch_size` isn't in the search space → KeyError.
   - `callbacks=args.call_back_monitor` — that's a string, not a callback list.
   - rank/save block is inside the trial loop; `args.results_folder` doesn't exist (it's `results_path`).
   - **Decide:** keep this manual file or delete it in favor of `tuning_driver_randomsearch.py`.

---

## B. Config ↔ parser

**`configs/training.txt` — nearly aligned now ✅**, one mismatch left:
- `--keras_path=models` (config) and `args.keras_path` (driver) vs **`--keras_files_path`** (parser).
  Three names for one thing — pick one (recommend `--keras_path` everywhere; shorter).

**`configs/search_space.txt` — heavily misaligned** (looks copied from WaterTemp's *keras-tuner* config,
but the manual tuner is sklearn-based). Args not defined in the parser → argparse will reject:
`--results_folder`, `--data_set`, `--max_trials`, `--executions_per_trial`, `--unit_list`,
`--activation_function_list`, `--layers_list`, `--tuner_objective`, `--activation_function` (renamed),
and `--input_window` is given 4 values but the parser arg is a single `int`.
Also `--metrics macro_f1` / `--tuner_objective macro_f1` aren't Keras-computable metrics.
→ This config doesn't match either tuner. If you keep the sklearn tuners, rebuild it around their
actual args (`n_iter`, `random_state`, window list, etc.).

---

## C. Structural

1. **Still no `__init__.py` anywhere**, and imports are now *inconsistent*: the drivers import
   `from src.data_prep...` (fine from repo root) but `tuning_driver.py` imports
   `from src.driver.training_driver` — **wrong**, drivers live in root `driver/`, so it should be
   `from driver.training_driver`. Pick one convention and make every import match.
2. **Entry points still empty:** `run_inference.py` (0 B), `run_tuning.py` (0 B),
   `run_training.py` (comment stub). Nothing wires parser → config → driver.
3. **No orchestration:** `training_driver` has no `__main__` and never calls inference/visualization;
   nothing chains train → infer → visualize.
4. `tuning_driver_randomsearch.py` is the **one driver that actually runs** end-to-end today.

---

## D. Per-file status

- **`parser.py`** — healthy. Remaining dead args to prune: `rotation_list`, `input_structure`,
  `kernel_regularizer`, `start_iteration`/`end_iteration`, `factor`, `min_lr`, `min_delta`.
- **`training_driver.py`** — close, but A1 + A2 block it; **still doesn't save the scaler**
  (`joblib.dump`) — inference/XAI need it.
- **`inference_driver.py`** — indentation fixed; still has dead imports and no `load_model`
  (relies on a caller passing `model`, which nothing does yet).
- **`visualization_driver.py`** — `loss_curve` good; `confusion_matrix` + `time_series` broken (A3/A4).
- **`tuning_driver.py`** — broken (A5); decide its fate.
- **`tuning_driver_randomsearch.py`** — ✅ runs; nested window-grid + RandomizedSearchCV + macro-F1 +
  `class_weight="balanced"`.
- **`explainability_driver.py`** — still pseudocode (expected).
- **`preparing_data.py`** — healthy; `usdm_severity` feature added/tested; `SOIL_PATH` placeholder.

---

## E. Methodology notes (keep in mind)

- **Save the scaler in training** (`joblib.dump`) — still the missing link for inference + XAI.
- **Class imbalance is the top issue.** `class_weight="balanced"` is now in the randomsearch tuner;
  carry it into `training_driver` too. D4 still absent from the fixed validation split → favor
  TimeSeriesSplit CV (already used in the randomsearch tuner).
- **Tuning didn't beat baseline (~0.34 macro-F1)** → confirms imbalance, not architecture, is the cap.
- **Ensemble idea** (from WaterTemp): soft-voting average of N seeded models → stability + a confidence
  signal; score with RPS/Brier (classification analogs of CRPS). Polish step, after imbalance.
- Single county, data capped at ~2024 (NDVI). Pooling counties is the main way to grow data.

---

## F. Housekeeping

- `README.txt` still says `preparingData` returns 6 arrays (it returns a 10-tuple).
- `AIML_notes.txt` appears empty.
- `TODO.txt` still has a pasted assistant response in it — trim to the task list.

---

## Suggested fix order

1. `training_driver`: fix `input_dim` (A1) + the `model.save` line (A2) + **add `joblib.dump(scaler)`**.
2. Decide `keras_path` vs `keras_files_path` and make parser/config/driver agree (B).
3. Fix `visualization_driver` `confusion_matrix`/`time_series` (A3/A4).
4. Settle the manual `tuning_driver.py` — fix or delete (A5); rebuild `search_space.txt` to match (B).
5. Fix import convention + add `__init__.py` (or commit to run-from-root) (C1).
6. Write the thin `run_*.py` entry points and the train→infer→viz orchestration (C2/C3).
7. Then modeling: carry `class_weight` into training, ensemble + RPS, county pooling.
