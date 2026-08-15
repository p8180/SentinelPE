# SentinelPE — Deployment Documentation

## Live Application

**Render URL:** https://sentinelpe.onrender.com/

The SentinelPE web application is deployed on Render and is publicly accessible.

## API Endpoints

### Health Check

**URL:** https://sentinelpe.onrender.com/health

The deployed health check confirms:

```json
{
  "feature_count": 26,
  "model_loaded": true,
  "model_type": "Pipeline",
  "status": "ok"
}
```

This verifies that the production service is running and that the trained preprocessing/model pipeline has loaded successfully.

### Prediction

**URL:** https://sentinelpe.onrender.com/predict

The `/predict` endpoint accepts a POST request containing the 26 expected features and returns:

- predicted class
- numeric prediction
- probability of goodware
- probability of malware

`/predict` is a POST endpoint; opening it directly in a browser with a GET request will return `405 Method Not Allowed`, which is expected.

## Model

- **Model:** Random Forest
- **Model version:** 1.0.0
- **Selection metric:** F1
- **Cross-validation:** Stratified 10-fold CV
- **Best CV F1:** 0.98496
- **Number of input features:** 26
- **Random state:** 42

### Selected hyperparameters

- `n_estimators`: 150
- `max_depth`: None
- `min_samples_split`: 2

## Final Hold-out Test Results

The selected Random Forest was evaluated on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.47% |
| Precision | 98.81% |
| Recall | 98.39% |
| F1 | 98.60% |
| ROC-AUC | 99.70% |

## Cross-Validation Model Comparison

The model-selection stage used stratified 10-fold cross-validation:

| Model | Mean Accuracy | Mean Precision | Mean Recall | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 98.35% | 98.80% | 98.19% | **98.50%** | **99.71%** |
| Decision Tree | 98.07% | 98.24% | 98.26% | 98.25% | 98.05% |
| SVM | 97.29% | 97.98% | 97.08% | 97.52% | 98.92% |
| Gradient Boosting | 96.70% | 97.42% | 96.55% | 96.99% | 99.21% |
| Logistic Regression | 96.71% | 97.88% | 96.10% | 96.98% | 98.78% |

Random Forest was selected because it achieved the strongest cross-validation F1 and ROC-AUC performance.

## Automated Testing

The API test suite completed successfully:

```text
5 passed
```

The tests cover the API's expected behaviour, including valid and invalid prediction requests.

## Live Inference Validation

### Goodware demonstration

The deployed `/predict` endpoint returned:

```json
{
  "label": "goodware",
  "prediction": 0,
  "probability_goodware": 0.9933333333333333,
  "probability_malware": 0.006666666666666667
}
```

**Result: PASS**

### Malware demonstration

The deployed `/predict` endpoint returned:

```json
{
  "label": "malware",
  "prediction": 1,
  "probability_goodware": 0.006666666666666667,
  "probability_malware": 0.9933333333333333
}
```

**Result: PASS**

## Deployment Architecture

```text
User / Browser
      |
      v
SentinelPE Web Interface
      |
      v
Flask REST API
      |
      v
Preprocessing + Random Forest Pipeline
      |
      v
26-feature representation
      |
      v
Goodware / Malware prediction
      |
      v
Class probabilities
```

The application is served using **Gunicorn** on Render.

The trained model is distributed separately from the Git repository as a GitHub Release asset because the model file exceeds GitHub's standard repository file-size limit.

## Source Repository

GitHub:

https://github.com/p8180/SentinelPE

## Deployment Status

**Status: LIVE AND VALIDATED**

The following have been successfully verified:

- GitHub source repository — PASS
- Render deployment — PASS
- Health endpoint — PASS
- Model loading — PASS
- Automated API tests — 5/5 PASS
- Live goodware inference — PASS
- Live malware inference — PASS

## Notes

This deployment is intended as a demonstration of an end-to-end machine-learning application. The reported test metrics are based on the project's held-out evaluation set and should not be interpreted as a guarantee of performance on unseen real-world malware beyond the evaluation data.
