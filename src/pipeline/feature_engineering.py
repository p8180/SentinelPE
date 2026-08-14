"""Feature engineering utilities."""

import pandas as pd

def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["MD5", "SHA1", "Name"] if c in df.columns]
    return df.drop(columns=cols)

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()
