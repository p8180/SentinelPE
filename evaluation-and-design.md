# SentinelPE — Evaluation and Design

## 1. Overview

SentinelPE is a binary machine-learning classifier designed to distinguish **goodware (benign PE samples)** from **malware** using the Brazilian malware dataset.

The final experiment used **50,758 labelled samples**:

- Goodware: **21,116 (41.61%)**
- Malware: **29,642 (58.39%)**

Labels were explicitly assigned during dataset assembly:

- `Label = 0` → goodware
- `Label = 1` → malware

The final production candidate is a **Random Forest classifier**.

## 2. Dataset ingestion and preparation

The malware source consists of the `malware-by-day` CSV collection. The directory contained many CSV files, including empty files.

During ingestion:

- Empty/invalid CSV files were skipped rather than deleted.
- **1,296** non-empty malware CSV files were processed.
- **1,501** empty/invalid malware CSV files were skipped.
- **29,642 malware rows** were loaded.
- Non-UTF-8 source text was handled using `latin-1` decoding.
- Goodware and malware records were combined into one labelled dataset.

No source files were modified or deleted.

### Label construction

The source malware records did not reliably provide the required binary target. Labels were therefore assigned from dataset provenance:

```text
Goodware → 0
Malware  → 1
```

Validation confirmed:

```text
Missing labels = 0
Total labelled samples = 50,758
```

## 3. Train/test design

A **stratified 80/20 train-test split** was used.

The test set was held out before model selection and hyperparameter tuning and was not used to select the final model.

The training portion was used for model comparison, stratified 10-fold cross-validation, and hyperparameter tuning. The final selected model was then evaluated once on the untouched hold-out test set.

## 4. Cross-validation methodology

Model selection used **Stratified 10-fold Cross-Validation**. Stratification preserves class proportions across folds and is appropriate for this binary classification task.

The principal model-selection metric was **F1**, balancing precision and recall. Additional metrics were Accuracy, Precision, Recall, F1, and ROC-AUC.

## 5. Candidate model comparison

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **98.35%** | **98.80%** | **98.19%** | **98.50%** | **99.71%** |
| Decision Tree | 98.07% | 98.24% | 98.26% | 98.25% | 98.05% |
| SVM | 97.29% | 97.98% | 97.08% | 97.52% | 98.92% |
| Gradient Boosting | 96.70% | 97.42% | 96.55% | 96.99% | 99.21% |
| Logistic Regression | 96.71% | 97.88% | 96.10% | 96.98% | 98.78% |

### Cross-validation stability

Random Forest achieved:

- Mean F1: **0.9849606**
- F1 standard deviation: **0.0009192**
- Mean accuracy: **0.9834966**
- Mean ROC-AUC: **0.9971413**

The small F1 standard deviation indicates consistent performance across folds.

## 6. Model selection and tuning

Random Forest was selected because it achieved the strongest overall cross-validation performance, including the highest accuracy, precision, F1 and ROC-AUC, with very high recall.

The selected model was tuned using 10-fold cross-validation.

Final configuration:

```text
Model: Random Forest
n_estimators: 150
max_depth: None
min_samples_split: 2
random_state: 42
```

Best CV F1:

**0.9849606**

## 7. Final hold-out test evaluation

The final Random Forest was evaluated on the previously untouched 20% hold-out test set.

| Metric | Hold-out result |
|---|---:|
| Accuracy | **98.47%** |
| Precision | **98.81%** |
| Recall | **98.39%** |
| F1 | **98.60%** |
| ROC-AUC | **99.70%** |

The final F1 score was **0.9860411**.

The difference between CV F1 (**0.9849606**) and hold-out F1 (**0.9860411**) is approximately **0.11 percentage points**, indicating no obvious degradation on held-out data.

## 8. Data preprocessing

The preprocessing workflow was:

```text
Raw datasets
    ↓
CSV validation / ingestion
    ↓
Skip empty or invalid source files
    ↓
Assign binary labels
    ↓
Combine goodware + malware
    ↓
Stratified 80/20 split
    ↓
Preprocessing within the training/CV pipeline
    ↓
Model training and selection
    ↓
Final hold-out evaluation
```

Preprocessing transformations were kept inside the sklearn pipeline so that transformations are learned from training data rather than from the complete dataset. This reduces the risk of preprocessing leakage.

## 9. Feature engineering

The training metadata records **26 input features before preprocessing**.

No artificial target-derived features were introduced.

Feature transformations remain inside the sklearn pipeline so that the same transformations can be applied during production inference.

### Feature-leakage consideration

The very high ROC-AUC values warrant caution. Feature definitions should be reviewed to ensure that file identifiers, dataset-provenance fields, timestamps, hashes, or other variables that indirectly encode the source class are not being used as predictive features.

The reported evaluation therefore demonstrates strong performance **on the supplied dataset and feature representation**. It should not automatically be interpreted as equivalent performance on an independently collected malware population.

## 10. Handling source-data irregularities

The original malware-by-day directory contained many empty CSV files.

Rather than deleting them manually, ingestion handles them explicitly:

- zero-byte files are skipped;
- unusable CSVs are skipped;
- successfully loaded files are concatenated;
- processed and skipped counts are reported.

The source also contained bytes that could not be decoded as UTF-8; `latin-1` decoding was used for the affected CSV collection.

## 11. SVM implementation decision

The initial SVM used `SVC(probability=True)`. This produced a scikit-learn deprecation warning and substantially increased training time because probability calibration was repeated during cross-validation.

The implementation was changed to **LinearSVC**.

This retains an SVM candidate while avoiding the expensive probability-calibration step. ROC-AUC is calculated from the classifier's decision scores.

The required stratified 10-fold CV methodology was retained.

## 12. Reproducibility

A fixed `random_state = 42` was used where supported.

The experiment records:

- model name;
- model version;
- training timestamp;
- CV fold count;
- selected hyperparameters;
- CV F1;
- feature count;
- final hold-out metrics.

The fitted model is saved as:

```text
models/best_model.pkl
```

## 13. Interpretation

Random Forest is the strongest of the five evaluated candidates for this dataset.

Its performance is consistent across CV and hold-out evaluation:

- CV F1: **98.50%**
- Hold-out F1: **98.60%**
- Hold-out ROC-AUC: **99.70%**

The close CV/hold-out F1 values are encouraging. However, the unusually high performance means independent validation is advisable before claiming equivalent performance on malware from other sources, collection periods, or distributions.

## 14. Final conclusion

The SentinelPE modelling experiment successfully completed:

1. Dataset ingestion and validation.
2. Binary label construction.
3. Stratified 80/20 hold-out splitting.
4. Stratified 10-fold cross-validation.
5. Benchmarking of five candidate classifiers.
6. Hyperparameter tuning.
7. Random Forest selection.
8. Final evaluation on the untouched hold-out test set.
9. Persistence of the final model artifact.

The final selected model is:

**Random Forest — 150 trees, unlimited maximum depth, minimum split size 2.**

Final hold-out performance:

**98.47% accuracy, 98.81% precision, 98.39% recall, 98.60% F1, and 99.70% ROC-AUC.**

These results provide the model-development basis for the production inference API, web application, automated testing, CI/CD pipeline, and Render deployment.
