from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import TinyBinaryClassifier

# --------------------------
# Pydantic schema for input
# --------------------------
class InputItem(BaseModel):
    x: list[float]


# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()


# --------------------------
# Load model at startup
# --------------------------
model = TinyBinaryClassifier()
model.load_state_dict(torch.load("tiny_model.pt", map_location="cpu"))
model.eval()


@app.get("/")
def home():
    return {"message": "Model API is running!"}


@app.post("/predict")
def predict(item: InputItem):
    # Convert input to tensor
    x = torch.tensor(item.x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        prob = model(x).item()

    pred_class = 1 if prob > 0.5 else 0

    return {
        "probability": prob,
        "prediction": pred_class
    }