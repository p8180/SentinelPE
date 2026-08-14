"""Merge datasets."""

import pandas as pd


def merge_datasets(goodware, malware):
    """
    Combine goodware and malware datasets and assign binary labels.

    Goodware = 0
    Malware  = 1
    """

    goodware = goodware.copy()
    malware = malware.copy()

    # Explicitly assign the target labels.
    goodware["Label"] = 0
    malware["Label"] = 1

    # Ensure both datasets have the same feature columns.
    common_columns = sorted(
        set(goodware.columns).intersection(malware.columns)
    )

    if "Label" not in common_columns:
        raise ValueError(
            "Label column could not be established in both datasets."
        )

    combined = pd.concat(
        [
            goodware[common_columns],
            malware[common_columns],
        ],
        ignore_index=True,
    )

    return combined