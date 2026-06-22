# Repo Audit — drought model

_A full pass over the repo. Grouped by severity. Line numbers are approximate and
will drift as you edit. "Blocking" = stops the code from running at all._

---

## A. Critical / blocking bugs (fix these first)

1. **`parser.py` never returns the parser.** `create_parser()` builds `parser` but has no
   `return parser` at the end → returns `None` → every caller crashes with
   `AttributeError: 'NoneType' has no attribute 'parse_args'`. **This blocks the entire
   config-driven path.** One line fix.

2. **`build_model()` is called with the wrong second argument.**
   `driver/training_driver.py` defines `build_model(input_dim, args)` but `main()` calls
   `build_model(x_train.shape[1], y_train.shape[1])` — it passes an **int** (`6`) where
   `args` is expected. Inside, `args.neurons` / `args.num_classes` will fail. Pass `args`.

3. **`callbacks=early` must be a list.** In `training_driver.main()`,
   `model.fit(..., callbacks=early)` passes a single callback object; Keras expects
   `callbacks=[early]`.

4. **`results_path` is a string but used as a Path.** Parser default is `'results'` (str),
   but `training_driver` calls `args.results_path.mkdir(...)` and `args.results_path / "..."`.
   A str has no `.mkdir`. Make the arg `type=Path` (or wrap with `Path(...)`).

5. **`visualization_driver.loss_curve()` references undefined names.** It uses `y_true`,
   `y_pred`, `acc`, `macro_f1`, and `pred_df`, none of which are passed in or defined in
   the function (they live in `inference`). As written it will `NameError`. The confusion
   matrix + timeseries blocks need those values passed in as arguments.

6. **`training_logger.py` uses `tf` without importing it.** `class TrainingLogger(tf.keras...)`
   → `NameError` the moment it's imported. Add `import tensorflow as tf`. (Not currently
   imported anywhere, but broken.)

---

## B. Config ↔ parser mismatches (block the config path once A.1 is fixed)

`configs/configs.txt` uses flags the parser doesn't define (argparse will reject them):

| Config flag | Problem | Fix |
|---|---|---|
| `--class_models` | not defined | parser has `--class_names` — rename in config |
| `--num_classes` | not defined | parser has `--num_output_neurons` — pick ONE name |
| `--hidden_activation` | not defined | parser has `--activation_function` — align |
| `--feature_cols` | **not defined at all** | add to parser (drivers + preparing_data need it) |
| `--scale=true` | `--scale` is `store_true` | use a bare `--scale` line (no `=true`) |
| `--optimizer=adam` | listed twice | remove the duplicate |
| `--v` | works only by abbreviation of `--verbose` | use `-v` |

Also referenced in code but mismatched:
- `training_driver` uses `args.monitor`; parser defines `--call_back_monitor` (no `--monitor`).
- `training_driver` uses `args.num_classes`, `args.hidden_activation`; parser doesn't define them.
- EarlyStopping uses `patience=args.lr_reducer_patience` — probably meant `early_stop_patience`.

**Root cause: the parser and config drifted apart.** Make the parser the single source of
truth, then make the config + drivers use those exact names.

---

## C. Structural / organization issues

1. **Two driver locations.** Real drivers live in root `driver/`; my pseudocode landed in
   `src/driver/` (`tuning_driver.py`, `explainability_driver.py`). There are now **two**
   `tuning_driver.py` files (a stub in `driver/`, pseudocode in `src/driver/`). Pick ONE
   location (recommend `driver/` to match your real ones) and move the pseudocode there.

2. **No `__init__.py` anywhere.** The `from src.data_prep...` / `from src.helper...` imports
   only work via implicit namespace packages **when run from the repo root**. The root
   `driver/` folder isn't a package either. Either commit to "always run from root with
   `python -m`", or add `__init__.py` files and a consistent package layout. Document it.

3. **Entry points are empty.** `scripts/run_training.py` is still a commented stub;
   `run_inference.py` and `run_tuning.py` are 0 bytes. Nothing wires parser → config →
   driver, so the config-driven pipeline cannot actually be launched yet.

4. **No orchestration.** `training_driver.main()` trains but never calls inference or
   visualization, and the run scripts don't chain them. Decide who calls whom.

---

## D. Per-file notes

**`driver/training_driver.py`**
- Does **not save the model or scaler** (`model.save(...)`, `joblib.dump(scaler, ...)`).
  Inference + explainability both depend on loading these → currently impossible. (See E.)
- Builds `model_datasplit` dict, good — but then mostly re-fetches via `.get(...)`; fine,
  just verbose.

**`driver/inference_driver.py`**
- Imports `Sequential/Input/Dense/EarlyStopping` but never uses them (dead imports).
- Never **loads** a `.keras` model (no `load_model`), despite the docstring. It receives a
  `model` arg, so something upstream must load + pass it.
- Indentation bug (lines ~39–41): the `pred_df[...] = ...` and `pred_df.to_csv(...)` are
  inside the `for` loop, so the CSV is rewritten every iteration and `to_csv` shouldn't be
  in the loop. De-indent `to_csv` out of the loop.

**`driver/visualization_driver.py`**
- Duplicate plotly imports; duplicated loss-figure creation (lines ~18–25).
- Function is named `loss_curve` but also does confusion matrix + timeseries — split or
  rename, and pass in `y_true`, `y_pred`, `pred_df`, `acc`, `macro_f1` (see A.5).

**`driver/tuning_driver.py`** — one-line stub; superseded by the `src/driver/` pseudocode.

**`src/data_prep/preparing_data.py`** — healthy. `SOIL_PATH` is a placeholder; `usdm_severity`
feature added and tested. (Minor: `usdm_severity` isn't in the NaN/contiguity guard, but it's
complete by construction from the label grid.)

**`parser.py`** — besides A.1/B: `--metrics` default is `[0]` (invalid Keras metric); still
carries dead WaterTemp args (`temperature_list`, `pred_atp_interval`, `input_structure`,
`tune_train_test`, `rotation_list`, `start/end_iteration`, `kernel_regularizer`, `min_delta`).
Prune so the config surface matches the model.

---

## E. Things to keep in mind (not bugs)

- **Save model + scaler in training.** Without `model.save()` + `joblib.dump(scaler)`, the
  inference and explainability drivers have nothing to load. This is the missing link
  between train → infer → explain.
- **Class imbalance is the top modeling issue.** D0/D1 are barely learned and **D4 has zero
  examples in validation** (artifact of the chronological split). Class weights + a
  validation strategy that includes every class are the highest-value fixes.
- **Baseline is ~0.42 accuracy / ~0.30 macro-F1** — a working scaffold, not a tuned model.
- **`usdm_severity` is a strong persistence feature.** Expect XAI to show the model leaning
  on it; report that honestly.
- **Data span is capped at ~2024** (NDVI ends 2024) and is **single-county**. Pooling
  counties is the main way to grow data + rare-class coverage.
- **Always evaluate with macro-F1 + confusion matrix**, never accuracy alone (imbalance).

---

## F. Housekeeping

- `README.txt` is slightly stale: says `preparingData` returns 6 arrays; it actually returns
  a 10-tuple (`+ dates + scaler`).
- `AIML_notes.txt` appears empty.
- `TODO.txt` has a pasted assistant response in it (lines ~7–23) — clutter; trim to the task list.

---

## Suggested fix order

1. `return parser` (A.1) — unblocks everything.
2. Align parser names ↔ config ↔ driver attribute names (A.2, B). Prune dead args.
3. Fix the driver bugs: `build_model(args)`, `callbacks=[early]`, `results_path` as Path,
   `monitor`/patience names (A.2–A.4).
4. Add **model + scaler saving** to training (E) — prerequisite for infer/XAI.
5. Fix `visualization_driver` to receive the values it needs (A.5);
   fix `inference_driver` indentation + add `load_model` (D).
6. Write the thin `scripts/run_*.py` entry points to wire parser → config → drivers (C.3).
7. Consolidate driver locations + decide packaging/`__init__.py` (C.1, C.2).
8. Then modeling work: class weights, validation split, tuning, XAI.
