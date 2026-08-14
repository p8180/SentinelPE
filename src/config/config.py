"""Central configuration for SentinelPE."""

from dataclasses import dataclass
from pathlib import Path

from .constants import RANDOM_STATE, TEST_SIZE, CV_FOLDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "datasets"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    logs_dir: Path = PROJECT_ROOT / "logs"

    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS

settings = Settings()
