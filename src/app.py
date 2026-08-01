# src/app.py
# FastAPI inference service for the RUL prediction model.
# Serves the CNN-LSTM model (your best performer) behind a REST endpoint.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch
import numpy as np
import joblib
import os

from phase5_model import CNN_LSTM_Model

app = FastAPI(
    title="RUL Prediction API",
    description="Predicts Remaining Useful Life (RUL) of a turbofan engine "
                "from the last 30 cycles of sensor readings (NASA CMAPSS FD001).",
    version="1.0.0"
)

WINDOW_SIZE = 30

# Resolve model/scaler paths relative to THIS file's location, not the
# current working directory -- so it works whether you run uvicorn from
# the project root or from inside src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # project root
SCALER_PATH = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'cnn_lstm_model.pth')

# Must match the exact column order used during training
# (op_settings + the 14 sensors that survived the constant-sensor filter)
FEATURE_COLUMNS = [
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]
NUM_FEATURES = len(FEATURE_COLUMNS)

# ---- Load model + scaler once at startup (not per-request, for speed) ----
scaler = joblib.load(SCALER_PATH)

model = CNN_LSTM_Model(num_features=NUM_FEATURES, window_size=WINDOW_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()


class CycleReading(BaseModel):
    """Raw sensor reading for a single cycle (one row of engine data)."""
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    # Exactly 30 cycles of raw (unscaled) sensor readings, oldest first
    cycles: List[CycleReading] = Field(
        ..., min_length=WINDOW_SIZE, max_length=WINDOW_SIZE,
        description=f"Exactly {WINDOW_SIZE} cycles of raw sensor readings, oldest to newest."
    )


class PredictionResponse(BaseModel):
    predicted_rul: float
    model_used: str = "CNN_LSTM"
    window_size: int = WINDOW_SIZE


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(request: PredictionRequest):
    try:
        # Convert the 30 cycle readings into a (30, 17) raw feature matrix,
        # preserving FEATURE_COLUMNS order (must match training exactly)
        raw_matrix = np.array([
            [getattr(cycle, col) for col in FEATURE_COLUMNS]
            for cycle in request.cycles
        ])  # shape: (30, 17)

        # Apply the SAME MinMax scaler fitted during training
        scaled_matrix = scaler.transform(raw_matrix)  # shape: (30, 17)

        # Add batch dimension -> (1, 30, 17)
        input_tensor = torch.tensor(scaled_matrix, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        # RUL can't be negative in practice, and was capped at 125 during
        # training, so clip the output to that same sane range
        prediction = max(0.0, min(prediction, 125.0))

        return PredictionResponse(predicted_rul=round(prediction, 1))

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "RUL Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict (POST)"
    }