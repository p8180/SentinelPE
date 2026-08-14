"""Factory for practical SentinelPE candidate classifiers."""
from __future__ import annotations
from typing import Dict
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from src.config.constants import RANDOM_STATE

class ModelFactory:
    @staticmethod
    def get_models() -> Dict[str, ClassifierMixin]:
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
            "SVM": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        }
