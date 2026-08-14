"""Model tuning, fitting, and artifact persistence for SentinelPE."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.config.constants import CV_FOLDS, RANDOM_STATE
from src.core.logger import get_logger
from src.models.model_factory import ModelFactory
from src.pipeline.preprocessing import build_preprocessor

logger = get_logger(__name__)

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Logistic Regression": {"model__C":[0.1,1.0,10.0], "model__solver":["liblinear"]},
    "Decision Tree": {"model__max_depth":[10,20,None], "model__min_samples_split":[2,5]},
    "Random Forest": {"model__n_estimators":[100,150], "model__max_depth":[None,20], "model__min_samples_split":[2,5]},
    "SVM": {"model__C":[0.1,1.0,10.0]},
    "Gradient Boosting": {"model__n_estimators":[100,150], "model__learning_rate":[0.05,0.1], "model__max_depth":[2,3]},
}

class ModelTrainer:
    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_STATE, param_grids=None):
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.random_state = random_state
        self.param_grids = param_grids or DEFAULT_PARAM_GRIDS
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_cv_score_ = None

    def tune_and_fit(self, X_train, y_train, model_name: str):
        models = ModelFactory.get_models()
        if model_name not in models:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(models)}")
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", models[model_name])])
        print(f"Starting 10-fold GridSearchCV: {model_name} ...")
        search = GridSearchCV(pipeline, self.param_grids[model_name], scoring="f1", cv=self.cv, n_jobs=-1, refit=True, return_train_score=False)
        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = dict(search.best_params_)
        self.best_cv_score_ = float(search.best_score_)
        logger.info("Best %s CV F1: %.6f", model_name, self.best_cv_score_)
        return self.best_estimator_

    def save_artifacts(self, model_path, metadata_path, model_name, feature_count):
        if self.best_estimator_ is None:
            raise RuntimeError("No fitted model is available to save.")
        model_path, metadata_path = Path(model_path), Path(metadata_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_estimator_, model_path)
        metadata = {
            "model_name": model_name, "model_version":"1.0.0",
            "training_timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "selection_metric":"f1", "cv_folds":self.cv.n_splits,
            "best_cv_f1":self.best_cv_score_, "best_parameters":self.best_params_,
            "feature_count_before_preprocessing":feature_count, "random_state":self.random_state
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
