from fastapi import APIRouter
from typing import List
import json
import joblib
import os

from wicket_api.wicket_schema import WicketInput
from wicket_api.wicket_service import predict_wicket

router = APIRouter()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "AI Wicket Probability Predictor"
MODEL_VERSION = "v1"
STUDENT_NAME = "Omkaar Shakti Sharma"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "wicket_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "feature_columns.json"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "metrics.json"
)

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except:
    model_loaded = False

try:
    with open(FEATURES_PATH, "r") as f:
        feature_columns = json.load(f)
except:
    feature_columns = []

try:
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
except:
    metrics = {}

# --------------------------------------------------
# ROOT
# --------------------------------------------------

@router.get("/")
def root():

    return {
        "api_name": MODEL_NAME,
        "student_name": STUDENT_NAME,
        "description":
        "Predicts the probability of a wicket "
        "falling on the next ball."
    }

# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@router.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

# --------------------------------------------------
# MODEL INFO
# --------------------------------------------------

@router.get("/model-info")
def model_info():

    return {

        "model_name": MODEL_NAME,

        "model_version": MODEL_VERSION,

        "model_type": "Logistic Regression",

        "target_variable":
        "target_is_wicket",

        "feature_columns":
        feature_columns,

        "metrics":
        metrics
    }

# --------------------------------------------------
# SINGLE PREDICTION
# --------------------------------------------------

@router.post("/wicket/predict")
def wicket_prediction(data: WicketInput):

    result = predict_wicket(data)

    return result

# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

@router.post("/wicket/predict-batch")
def wicket_prediction_batch(
    data: List[WicketInput]
):

    predictions = []

    for row in data:

        predictions.append(
            predict_wicket(row)
        )

    return {
        "count": len(predictions),
        "predictions": predictions
    }