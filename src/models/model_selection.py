"""Stratified cross-validation model benchmarking."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src.config.constants import CV_FOLDS, RANDOM_STATE
from src.pipeline.preprocessing import build_preprocessor
from .model_factory import ModelFactory

class ModelSelector:
    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE) -> None:
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def benchmark(self, X: pd.DataFrame, y: pd.Series, output_path=None):
        scoring = {"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}
        rows = []
        for name, estimator in ModelFactory.get_models().items():
            print(f"Running {self.cv.n_splits}-fold CV: {name} ...")
            pipeline = Pipeline([("preprocessor", build_preprocessor(X)), ("model", estimator)])
            scores = cross_validate(pipeline, X, y, cv=self.cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            rows.append({
                "model": name,
                "mean_accuracy": scores["test_accuracy"].mean(),
                "std_accuracy": scores["test_accuracy"].std(),
                "mean_precision": scores["test_precision"].mean(),
                "mean_recall": scores["test_recall"].mean(),
                "mean_f1": scores["test_f1"].mean(),
                "std_f1": scores["test_f1"].std(),
                "mean_roc_auc": scores["test_roc_auc"].mean(),
            })
        results = pd.DataFrame(rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)
        best_model = str(results.loc[0, "model"])
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
        return results, best_model
