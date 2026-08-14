"""Dataset validation helpers."""

import pandas as pd

REQUIRED_LABEL = "Label"

def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")
    if REQUIRED_LABEL not in df.columns:
        raise ValueError("Missing Label column.")
