# Dockerized Model API

FastAPI service that serves a tiny PyTorch binary classifier, ships in a slim Docker image, and is ready for Cloud Run.

## What’s inside
- FastAPI app with input validation, `/predict` and `/health` endpoints (`app/main.py`).
- Minimal PyTorch model architecture (`app/model.py`) and training script with MLflow logging (`app/train.py`).
- Container build ready for Cloud Run (`Dockerfile`) plus CI/CD workflow (`.github/workflows/deploy.yml`).
- Runtime and training dependency pins (`app/requirements.txt`, `app/requirements-dev.txt`).

## Project structure
```
.
├── app/
│   ├── main.py               # FastAPI app + model loading
│   ├── model.py              # TinyBinaryClassifier definition
│   ├── tiny_model.pt         # Saved weights (used at runtime)
│   ├── train.py              # Sample training pipeline (MLflow logged)
│   ├── requirements.txt      # Runtime dependencies
│   └── requirements-dev.txt  # Runtime + training deps (includes MLflow)
├── Dockerfile                # Container image (ports 8000)
└── .github/workflows/deploy.yml  # Cloud Run deploy workflow
```

## Prerequisites
- Python 3.12
- Docker (for containerized runs)
- A GCP project with Cloud Run + Artifact Registry (for the GitHub Action)

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000` for a ping message or `http://localhost:8000/docs` for Swagger UI.

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x": [0.1, -0.2, 0.3, 0.4]}'
```

## Training a new model
```bash
pip install -r app/requirements-dev.txt
python app/train.py
```
This logs metrics to MLflow (local files) and overwrites `app/tiny_model.pt`, which the API loads at startup.

## Docker
```bash
docker build -t model-api:latest .
docker run -p 8000:8000 model-api:latest
```

## Deploy (GitHub Actions → Cloud Run)
The workflow at `.github/workflows/deploy.yml`:
1) Authenticates with GCP using `GCLOUD_SERVICE_KEY`.
2) Builds and pushes `model-api:latest` to Artifact Registry `<REGION>-docker.pkg.dev/<PROJECT>/<REPOSITORY_NAME>/model-api`.
3) Deploys to Cloud Run service defined by `CLOUD_RUN_SERVICE` in region `GCP_REGION` with port 8000.

Configure these GitHub secrets:
- `GCLOUD_SERVICE_KEY` (service account JSON)
- `GCP_PROJECT_ID`
- `GCP_REGION`
- `REPOSITORY_NAME` (Artifact Registry repo)
- `CLOUD_RUN_SERVICE`

## Notes
- `tiny_model.pt` is checked in for easy demos; retrain if you change the architecture.
- `mlruns/` is ignored by git to keep the repo clean; point MLflow to a server or artifact store in production.
