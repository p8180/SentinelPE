# PR #7 — End-to-End Training Runner

## Purpose

`scripts/train.py` runs the complete SentinelPE training workflow in the
required order.

## Important data-leakage rule

The 20% stratified hold-out test set is created immediately after loading
and basic dataset cleaning. It is never passed to model selection or
hyperparameter tuning.

## Run

From the SentinelPE project root:

```bash
python scripts/train.py ^
  --goodware "datasets/raw/goodware/goodware.csv" ^
  --malware-dir "datasets/raw/malware-by-day"
```

On macOS/Linux, use `\` instead of `^` for line continuation, or place
the command on one line.

## Expected artifacts

```text
models/
  best_model.pkl

outputs/
  data_split_summary.csv
  model_benchmark_results.csv
  model_metadata.json
  metrics/
    test_metrics.csv
    classification_report.csv
    confusion_matrix.png
    final_results_summary.json
```
