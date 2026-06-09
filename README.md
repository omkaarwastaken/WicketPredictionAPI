# AI Wicket Probability Prediction

A machine learning project that predicts the probability of a wicket falling on the next ball of a cricket match.

The project combines a trained Logistic Regression model, a FastAPI backend, and a Streamlit dashboard to provide real-time cricket analytics.

---

## Project Overview

The objective of this project is to estimate wicket probability using the current match situation and the batter-bowler matchup.

The model considers factors such as:

- Match pressure
- Recent scoring pattern
- Current and required run rate
- Batter attributes
- Bowler attributes
- Over and ball information

The prediction is returned as a probability along with a simple threat label.

---

## Features

- Machine learning based wicket prediction
- FastAPI backend
- Streamlit dashboard
- Player selection by name
- Automatic player attribute lookup
- Batch prediction support
- Pre-trained `.pkl` model
- No retraining during API requests

---

## Project Structure

```
.
├── dashboard.py
├── main.py
├── requirements.txt
├── wicket_api/
│   ├── wicket_router.py
│   ├── wicket_schema.py
│   └── wicket_service.py
│
├── model/
│   ├── wicket_model.pkl
│   ├── feature_columns.json
│   ├── metrics.json
│   └── sample_input.json
│
├── model_training/
│   └── train_model.py
│
└── README.md
```

---

## Model

**Algorithm:** Logistic Regression

**Prediction Type:** Binary Classification

**Target Variable:**

- 0 → No wicket
- 1 → Wicket

The model is trained offline and saved as a `.pkl` file. The API only loads the saved model and never retrains it during requests.

---

## API Endpoints

### GET /

Returns basic API information.

### GET /health

Returns the current API status.

### GET /model-info

Returns information about the trained model.

### POST /wicket/predict

Returns a prediction for a single input.

### POST /wicket/predict-batch

Returns predictions for multiple inputs.

---

## Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

Swagger documentation will be available at:

```
http://127.0.0.1:8000/docs
```

---

## Running the Dashboard

Start Streamlit:

```bash
streamlit run dashboard.py
```

---

## Dashboard

The Streamlit dashboard allows users to:

- Select a batter
- Select a bowler
- View player roles
- Enter match information
- Generate wicket probability predictions

Player data is loaded automatically from the cricket dataset hosted on GitHub.

---

## Example Prediction Response

```json
{
    "prediction": 71.0,
    "prediction_label": "High Threat",
    "model_name": "AI Wicket Probability Predictor",
    "model_version": "v1",
    "explanation": "The current match situation and player matchup suggest a high probability of a wicket."
}
```

---

## Model Performance

The model was evaluated using standard classification metrics.

| Metric | Score |
|----------|--------|
| ROC AUC | 0.78 |
| Precision | 0.69 |
| Recall | 0.72 |
| F1 Score | 0.70 |

---

## Future Improvements

Potential improvements include:

- Weather and pitch data
- Recent player form
- Advanced machine learning models
- Win probability prediction
- Multi-ball sequence prediction

---

## Technologies Used

- Python
- FastAPI
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## Author

Omkaar Shakti Sharma and Nethra Vinod

