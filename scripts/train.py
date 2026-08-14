from __future__ import annotations

"""Reproducible end-to-end SentinelPE training runner.

Workflow:
1. Load labelled goodware/malware data.
2. Remove duplicate rows and identifier columns.
3. Create the stratified 80/20 hold-out split FIRST.
4. Benchmark candidate models on training data only.
5. Tune the selected model with stratified 10-fold CV.
6. Persist the fitted pipeline and metadata.
7. Evaluate ONCE on the untouched 20% test set.
"""
"""Reproducible end-to-end SentinelPE training runner."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.constants import RANDOM_STATE
from src.data.data_loader import load_goodware, load_malware
from src.data.merger import merge_datasets
from src.pipeline.feature_engineering import (
    remove_duplicate_rows,
    remove_identifier_columns,
)
from src.models.model_selection import ModelSelector
from src.services.training import ModelTrainer
from src.services.evaluation import ModelEvaluator, save_evaluation_summary


def build_dataset(goodware_path: str, malware_dir: str) -> pd.DataFrame:
    """Load and combine the labelled source datasets."""
    goodware = load_goodware(goodware_path)
    malware = load_malware(malware_dir)
    return merge_datasets(goodware, malware)


def run_training(
    goodware_path: str,
    malware_dir: str,
    output_dir: str = "outputs",
    model_dir: str = "models",
) -> None:
    """Execute the complete reproducible training workflow."""
    output = Path(output_dir)
    model_path = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    df = build_dataset(goodware_path, malware_dir)
    df = remove_duplicate_rows(df)

    if "Label" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'Label' target column."
        )

    X = remove_identifier_columns(df.drop(columns=["Label"]))
    y = df["Label"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            f"Expected binary target with 2 classes; found {y.unique()}."
        )

    # CRITICAL: create the hold-out set BEFORE model selection/tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Persist split sizes for reproducibility/audit purposes.
    pd.DataFrame([{
        "total_rows": len(df),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_fraction": len(X_test) / len(df),
        "training_class_0": int((y_train == 0).sum()),
        "training_class_1": int((y_train == 1).sum()),
        "test_class_0": int((y_test == 0).sum()),
        "test_class_1": int((y_test == 1).sum()),
    }]).to_csv(output / "data_split_summary.csv", index=False)

    # CV/model selection uses TRAINING DATA ONLY.
    selector = ModelSelector()
    benchmark_path = output / "model_benchmark_results.csv"
    benchmark_results, best_model = selector.benchmark(
        X_train,
        y_train,
        output_path=benchmark_path,
    )

    print("\nModel benchmark:")
    print(benchmark_results.to_string(index=False))
    print(f"\nSelected model: {best_model}")

    # Hyperparameter tuning also uses TRAINING DATA ONLY.
    trainer = ModelTrainer()
    trainer.tune_and_fit(
        X_train,
        y_train,
        model_name=best_model,
    )

    trained_model_path = model_path / "best_model.pkl"
    metadata_path = output / "model_metadata.json"

    trainer.save_artifacts(
        model_path=trained_model_path,
        metadata_path=metadata_path,
        model_name=best_model,
        feature_count=X_train.shape[1],
    )

    # FINAL evaluation: the test set is used only here.
    evaluator = ModelEvaluator.from_file(trained_model_path)
    test_metrics = evaluator.evaluate(
        X_test,
        y_test,
        output_dir=output / "metrics",
    )

    save_evaluation_summary(
        metrics=test_metrics,
        model_metadata_path=metadata_path,
        output_path=output / "metrics" / "final_results_summary.json",
    )

    print("\nFinal hold-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nTraining workflow completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate SentinelPE."
    )
    parser.add_argument("--goodware", required=True)
    parser.add_argument("--malware-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        goodware_path=args.goodware,
        malware_dir=args.malware_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
