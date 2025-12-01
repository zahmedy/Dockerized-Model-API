from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist

from model import TinyBinaryClassifier


class InputItem(BaseModel):
    """Request payload for a single prediction."""

    # Expect exactly 4 numeric features (matches TinyBinaryClassifier input)
    x: Annotated[
        conlist(float, min_items=4, max_items=4),
        Field(..., description="Array of 4 feature values"),
    ]


class PredictionResponse(BaseModel):
    probability: float
    prediction: int


app = FastAPI(title="Tiny Binary Classifier API", version="1.0.0")

MODEL_PATH = Path(__file__).resolve().parent / "tiny_model.pt"
model = TinyBinaryClassifier()
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


@app.get("/")
def home():
    return {"message": "Model API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(item: InputItem):
    x = torch.tensor(item.x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    if not 0.0 <= prob <= 1.0:
        raise HTTPException(status_code=500, detail="Model output out of bounds")

    pred_class = 1 if prob > 0.5 else 0
    return {"probability": prob, "prediction": pred_class}
