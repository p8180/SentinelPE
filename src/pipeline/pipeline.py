"""Pipeline creation."""

from sklearn.pipeline import Pipeline
from .preprocessing import build_preprocessor

def create_pipeline(model, X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", model)
    ])
