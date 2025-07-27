# MLOps Iris Classification Pipeline Summary

## Tools Used
- **GitHub**: Source control and CI/CDs
- **MLflow**: Experiment tracking
- **FastAPI**: REST API for model inference
- **Docker**: Containerization
- **GitHub Actions**: CI/CD automation
- **Logging**: Incoming requests & outputs

## Workflow
1. Dataset is saved locally in `data/`
2. Models trained in `src/train.py` with MLflow tracking
3. Best model saved to `models/best_model.pkl`
4. API built using FastAPI (`api/app.py`)
5. Dockerized and deployed using GitHub Actions

## CI/CD
- Every push triggers:
  - Python setup
  - Dependencies installation
  - Docker build & push to Docker Hub

## Monitoring
- `/metrics` endpoint exposes Prometheus metrics
- Logs written to `logs/prediction.log` 