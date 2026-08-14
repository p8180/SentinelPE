"""Dataset loading utilities."""

from pathlib import Path
import glob
import pandas as pd

def load_goodware(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Label"] = 0
    return df

def load_malware(malware_dir):
    """
    Load all non-empty malware CSV files from malware-by-day.

    Empty CSV files are skipped rather than causing the entire pipeline
    to fail.
    """
    malware_dir = Path(malware_dir)

    if not malware_dir.exists():
        raise FileNotFoundError(
            f"Malware directory not found: {malware_dir}"
        )

    frames = []
    skipped_files = []

    csv_files = sorted(malware_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {malware_dir}"
        )

    for csv_file in csv_files:

        # Skip genuinely empty files
        if csv_file.stat().st_size == 0:
            skipped_files.append(str(csv_file))
            continue

        try:
            df = pd.read_csv(
		csv_file,
    		encoding="latin-1"
	    )

            # Skip files with no columns
            if df.empty or len(df.columns) == 0:
                skipped_files.append(str(csv_file))
                continue

            frames.append(df)

        except pd.errors.EmptyDataError:
            skipped_files.append(str(csv_file))

    if not frames:
        raise ValueError(
            "No usable malware CSV files were found."
        )

    malware = pd.concat(frames, ignore_index=True)

    print(f"Loaded malware rows: {len(malware):,}")
    print(f"Malware CSV files processed: {len(frames):,}")
    print(f"Empty/invalid files skipped: {len(skipped_files):,}")

    return malware
