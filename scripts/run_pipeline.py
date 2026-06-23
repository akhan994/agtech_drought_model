"""
run_pipeline.py - orchestrator for the full train -> infer -> visualize chain.

Parses the config once and runs the three drivers in order. Because each step
hands off through disk (training writes model+scaler+history; inference writes
metrics + test_predictions.csv; visualization reads those), the steps stay
INDEPENDENTLY runnable -- this script just runs them back-to-back for convenience.

Run from the repo root:
    python -m scripts.run_pipeline @configs/training.txt
"""

from datetime import datetime
from pathlib import Path

from src.helper.parser import create_parser
from driver.training_driver import main as train_model
from driver.inference_driver import inference
from driver.visualization_driver import loss_curve, plot_confusion_matrix, time_series


def run(args):
    print("\nTraining model...")
    train_model(args)        # trains, saves model.keras + scaler.joblib + history.csv

    print("\nInferencing...")
    inference(args)          # loads latest run, writes metrics.txt + test_predictions.csv

    print("\nVisualizing...")
    loss_curve(args)         # all three read the artifacts the steps above wrote
    plot_confusion_matrix(args)
    time_series(args)

    print("\n=== DONE ===")


if __name__ == "__main__":
    args = create_parser().parse_args()   # pass @configs/training.txt on the CLI

    # give each run its own results folder: results/training_<timestamp>/
    # set ONCE here so train + infer + visualize all write into the same folder.
    args.results_path = Path(args.results_path) / f"training_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.results_path.mkdir(parents=True, exist_ok=True)
    print(f"results -> {args.results_path}")

    run(args)
