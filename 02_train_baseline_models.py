"""
Step 3a: Baseline Models
HR Analytics Employee Attrition & Performance
--------------------------------------------------------
Trains BASELINE (default hyperparameter) versions of:
  1. Logistic Regression
  2. Random Forest Classifier
  (3. XGBoost baseline is trained separately in 04_train_xgboost_LOCAL.py)

Evaluates on the validation set. These numbers are the "floor" that
hyperparameter tuning must beat to be worth reporting.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
val = pd.read_csv(f"{DATA_DIR}/val.csv")

X_train, y_train = train.drop(columns=["Attrition"]), train["Attrition"]
X_val, y_val = val.drop(columns=["Attrition"]), val["Attrition"]

# Logistic Regression needs scaled features; tree models don't.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")


def evaluate(model_name, y_true, y_pred, y_proba):
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


results = []

# ---------------- Logistic Regression (baseline) ----------------
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train_scaled, y_train)
pred = lr_baseline.predict(X_val_scaled)
proba = lr_baseline.predict_proba(X_val_scaled)[:, 1]
res = evaluate("Logistic Regression (baseline)", y_val, pred, proba)
results.append(res)
joblib.dump(lr_baseline, f"{MODEL_DIR}/lr_baseline.joblib")

# ---------------- Random Forest (baseline) ----------------
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)  # trees don't need scaling
pred = rf_baseline.predict(X_val)
proba = rf_baseline.predict_proba(X_val)[:, 1]
res = evaluate("Random Forest (baseline)", y_val, pred, proba)
results.append(res)
joblib.dump(rf_baseline, f"{MODEL_DIR}/rf_baseline.joblib")

with open(f"{RESULTS_DIR}/baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
