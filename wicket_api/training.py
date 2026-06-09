import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib

print("Starting")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("ball_training_features.csv")

print("CSV loaded")
print("Shape:", df.shape)
# --------------------------------------------------
# Load Supporting Tables
# --------------------------------------------------

ratings = pd.read_csv("player_attributes.csv")
players = pd.read_csv("players.csv")

# --------------------------------------------------
# Batter Ratings
# --------------------------------------------------

batter_ratings = ratings[
    [
        "player_id",
        "batting_rating",
        "aggression",
        "consistency",
        "pressure_handling"
    ]
].copy()

batter_ratings.columns = [
    "striker_id",
    "batter_rating",
    "batter_aggression",
    "batter_consistency",
    "batter_pressure"
]

df = df.merge(
    batter_ratings,
    on="striker_id",
    how="left"
)

# --------------------------------------------------
# Bowler Ratings
# --------------------------------------------------

bowler_ratings = ratings[
    [
        "player_id",
        "bowling_rating",
        "death_overs_skill",
        "pressure_handling"
    ]
].copy()

bowler_ratings.columns = [
    "bowler_id",
    "bowler_rating",
    "bowler_death_skill",
    "bowler_pressure"
]

df = df.merge(
    bowler_ratings,
    on="bowler_id",
    how="left"
)

# --------------------------------------------------
# Batter Hand
# --------------------------------------------------

batter_info = players[
    [
        "id",
        "batting_hand"
    ]
].copy()

batter_info.columns = [
    "striker_id",
    "batting_hand"
]

df = df.merge(
    batter_info,
    on="striker_id",
    how="left"
)

df["left_handed"] = (
    df["batting_hand"] == "Left"
).astype(int)

# --------------------------------------------------
# Derived Features
# --------------------------------------------------

df["rating_gap"] = (
    df["bowler_rating"]
    - df["batter_rating"]
)

df["pressure_matchup"] = (
    df["bowler_pressure"]   
    - df["batter_pressure"]
)

df["aggression_vs_skill"] = (
    df["batter_aggression"]
    * df["bowler_death_skill"]
)

# --------------------------------------------------
# Fill Missing
# --------------------------------------------------

df = df.fillna(0)

# --------------------------------------------------
# TARGET
# --------------------------------------------------

y = df["target_is_wicket"]

print("\nTarget Distribution")
print(y.value_counts())
print(y.value_counts(normalize=True))

# --------------------------------------------------
# FEATURES
# --------------------------------------------------

feature_columns = [

    "expected_wicket_probability",
    "pressure_index",
    "shot_risk_score",
    "batter_control_score",
    "current_run_rate",
    "required_run_rate",
    "over_number",
    "ball_number",

    "batter_rating",
    "batter_aggression",
    "batter_consistency",
    "batter_pressure",

    "bowler_rating",
    "bowler_death_skill",
    "bowler_pressure",

    "rating_gap",
    "pressure_matchup",
    "aggression_vs_skill",

    "left_handed"
]

X = df[feature_columns].copy()

X = X.fillna(0)

print("\nFeatures Used")

for col in feature_columns:
    print(col)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain test split done")

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    C=0.5
)
    

print("Starting training")

model.fit(
    X_train,
    y_train
)

print("Training complete")

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

pred_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# THRESHOLD SEARCH
# --------------------------------------------------

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.30, 0.81, 0.05):

    preds = (pred_prob >= threshold).astype(int)

    precision = precision_score(
        y_test,
        preds,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        preds,
        zero_division=0
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# --------------------------------------------------
# FINAL EVALUATION
# --------------------------------------------------

pred_class = (
    pred_prob >= best_threshold
).astype(int)

precision = precision_score(
    y_test,
    pred_class,
    zero_division=0
)

recall = recall_score(
    y_test,
    pred_class,
    zero_division=0
)

f1 = f1_score(
    y_test,
    pred_class,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    pred_prob
)

cm = confusion_matrix(
    y_test,
    pred_class
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nRESULTS")
print("-" * 40)

print("Best Threshold:", best_threshold)
print("ROC AUC:", roc_auc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nConfusion Matrix")
print(cm)

# --------------------------------------------------
# PROBABILITY SEPARATION
# --------------------------------------------------

wicket_probs = pred_prob[y_test == 1]
non_wicket_probs = pred_prob[y_test == 0]

print("\nProbability Separation")

print(
    "Average wicket probability:",
    wicket_probs.mean()
)

print(
    "Average non-wicket probability:",
    non_wicket_probs.mean()
)

# --------------------------------------------------
# FEATURE COEFFICIENTS
# --------------------------------------------------

print("\nFeature Coefficients")
print("-" * 40)

for feature, coef in zip(
    feature_columns,
    model.coef_[0]
):
    print(
        f"{feature}: {coef:.4f}"
    )

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "threshold": best_threshold,
        "features": feature_columns
    },
    "wicket_model.pkl"
)

print("\nModel saved.")