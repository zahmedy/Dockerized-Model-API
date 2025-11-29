# Dockerized-Model-API
FastAPI that loads a trained ML model, exposes /predict, and runs fully inside Docker.

```docker-model-api/
│
├── app/
│   ├── main.py           # FastAPI app
│   ├── model.joblib      # Saved model
│   └── requirements.txt  # pip deps
│
├── Dockerfile            # Build container image
└── README.md```


