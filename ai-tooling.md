# AI Tooling — SentinelPE

## 1. Purpose

AI-assisted development was used throughout the SentinelPE project to support planning, implementation, debugging, documentation, testing and deployment.

The AI tooling supported the development process; it did not replace the underlying machine-learning workflow or the validation of the deployed system.

## 2. Primary AI Tool

### ChatGPT

ChatGPT was used as an AI development and project-support assistant for:

- Project planning and decomposition
- Python and Flask implementation guidance
- Machine-learning pipeline design and refinement
- Code generation and code review
- Debugging Python, Git and deployment issues
- Test development and interpretation
- Git/GitHub workflow guidance
- Render deployment configuration
- API testing guidance
- Documentation generation
- Presentation and demonstration planning

## 3. Areas Where AI Assistance Was Used

### Machine-learning workflow

AI assistance was used to help structure and refine:

- Dataset loading and validation
- Exploratory data analysis planning
- Feature preprocessing
- Stratified train/test splitting
- Stratified 10-fold cross-validation
- Model comparison
- Hyperparameter tuning
- Hold-out evaluation
- Model persistence

The final model-selection and evaluation results were obtained by executing the project code and tests rather than relying on AI-generated claims.

### Application development

AI assistance supported development of:

- Flask API structure
- `/health` endpoint
- `/predict` endpoint
- Input validation
- Model-loading logic
- HTML/CSS web interface
- Application configuration
- Gunicorn deployment configuration

### Testing

AI assistance was used to:

- Interpret test failures
- Suggest debugging steps
- Construct API test requests
- Validate expected API behaviour
- Troubleshoot deployment issues

The project's automated test suite ultimately returned:

```text
5 passed
```

### Git and GitHub

AI assistance supported:

- Repository initialization
- Branch management
- `.gitignore` configuration
- Git staging and commits
- Remote configuration
- Resolving GitHub authentication/account issues
- Handling GitHub's 100 MB file-size restriction
- Removing the large model artifact from repository history
- Publishing the model as a GitHub Release asset

### Deployment

AI assistance supported the Render deployment process, including:

- `render.yaml` configuration
- Build/start command configuration
- Model artifact retrieval during deployment
- Gunicorn configuration
- Troubleshooting deployment failures
- Live `/health` testing
- Live `/predict` testing

## 4. Human Verification

AI-generated suggestions were treated as development assistance rather than authoritative results.

The following were independently executed and verified in the project environment:

- Python test suite
- Model training/evaluation workflow
- Git commits and repository state
- GitHub push
- Render deployment
- `/health` endpoint
- Live goodware inference
- Live malware inference

The deployed API successfully classified both demonstration samples:

- Goodware: 99.33% predicted goodware
- Malware: 99.33% predicted malware

## 5. Important AI-Assisted Development Principle

AI assistance accelerated implementation and troubleshooting, but project decisions and final claims were based on executable project artefacts and observed outputs.

In particular, model performance figures, test results and deployment status were not accepted merely because an AI assistant suggested them; they were verified through execution.

## 6. Reproducibility

Key project artefacts include:

- `README_PR7.md`
- `requirements.txt`
- `requirements-api.txt`
- `render.yaml`
- `src/`
- `scripts/`
- `tests/`
- `data/demo_samples.json`
- `models/best_model.pkl` / released model asset
- `deployed.md`

## 7. AI Tooling Summary

| Activity | AI assistance | Human/execution verification |
|---|---|---|
| Project planning | Yes | Yes |
| Code development | Yes | Yes |
| ML pipeline design | Yes | Yes |
| Model selection | Assisted | Executed and evaluated |
| Testing | Yes | Executed |
| Debugging | Yes | Executed |
| Git/GitHub | Yes | Executed |
| Deployment | Yes | Executed |
| API validation | Yes | Live-tested |
| Documentation | Yes | Reviewed |
| Presentation preparation | Yes | Reviewed |

## 8. Disclosure Statement

SentinelPE was developed with AI-assisted programming and documentation support. AI was used as a productivity and reasoning aid across the development lifecycle, while the executable code, model outputs, tests and deployment behaviour were validated through actual project execution.
