"""Final hold-out test evaluation for SentinelPE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate a fitted production pipeline on the untouched test set."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def from_file(cls, model_path: str | Path) -> "ModelEvaluator":
        """Load a persisted fitted pipeline."""
        return cls(joblib.load(model_path))

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_dir: str | Path = "outputs/metrics",
    ) -> dict[str, float]:
        """Evaluate once on the held-out test set.

        X_test and y_test must be the untouched 20% hold-out partition.
        They must not have been used for model selection or tuning.
        """
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same length.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = self.model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(
                roc_auc_score(y_test, probabilities)
            )

        pd.DataFrame([metrics]).to_csv(
            output_dir / "test_metrics.csv", index=False
        )

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            output_dir / "classification_report.csv"
        )

        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Goodware", "Malware"],
        )
        disp.plot()
        plt.title("SentinelPE — Hold-out Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            output_dir / "confusion_matrix.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        logger.info("Final hold-out evaluation completed.")
        logger.info("Test metrics: %s", metrics)

        return metrics


def save_evaluation_summary(
    metrics: dict[str, float],
    model_metadata_path: str | Path,
    output_path: str | Path = "outputs/metrics/final_results_summary.json",
) -> None:
    """Combine model metadata and final test metrics into one summary."""
    metadata_path = Path(model_metadata_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

    summary = {
        "model": metadata,
        "holdout_test_metrics": metrics,
    }

    output.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
