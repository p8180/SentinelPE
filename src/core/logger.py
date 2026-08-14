"""Logging utilities for SentinelPE."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logfile = RotatingFileHandler(
        LOG_DIR / "sentinelpe.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)
    logger.propagate = False
    return logger
