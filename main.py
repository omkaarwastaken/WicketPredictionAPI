from fastapi import FastAPI
import json
import joblib
import os

from wicket_api.wicket_router import router as wicket_router

app = FastAPI(
    title="AI Wicket Probability Prediction API",
    version="1.0.0",
    description="Predicts the probability of a wicket falling on the next ball."
)

# --------------------------------------------------
# LOAD MODEL INFORMATION
# --------------------------------------------------

MODEL_NAME = "AI Wicket Probability Predictor"
MODEL_VERSION = "v1"
STUDENT_NAME = "Omkaar Shakti Sharma"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Safe loading

try:
    model_bundle = joblib.load(MODEL_PATH)
except:
    model_bundle = None

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
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "api_name": MODEL_NAME,
        "student_name": STUDENT_NAME,
        "description":
        "Predicts the probability of a wicket "
        "falling on the next ball."
    }

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model_bundle is not None
    }

# --------------------------------------------------
# MODEL INFO
# --------------------------------------------------

@app.get("/model-info")
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
# EXISTING ROUTER
# --------------------------------------------------

app.include_router(
    wicket_router
)