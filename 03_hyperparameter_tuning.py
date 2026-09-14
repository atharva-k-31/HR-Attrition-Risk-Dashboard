"""
Step 3b: Hyperparameter Tuning
HR Analytics Employee Attrition & Performance
--------------------------------------------------------
Baselines above optimize implicitly for accuracy and under-catch leavers
(low recall). Here we tune with class_weight='balanced' in the search
space and score on F1 (harmonic mean of precision/recall) so the search
doesn't just chase accuracy on the majority class.
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

train = pd.read_csv(f"{DATA_DIR}/train.csv")
val = pd.read_csv(f"{DATA_DIR}/val.csv")

X_train, y_train = train.drop(columns=["Attrition"]), train["Attrition"]
X_val, y_val = val.drop(columns=["Attrition"]), val["Attrition"]

scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)


def evaluate(model_name, y_true, y_pred, y_proba, best_params=None):
    return {
        "model": model_name,
        "best_params": best_params,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


results = []

# ---------------- Logistic Regression (tuned) ----------------
lr_param_dist = {
    "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"],
    "solver": ["lbfgs", "liblinear"],
}
lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    param_distributions=lr_param_dist,
    n_iter=12, scoring="f1", cv=3, random_state=42, n_jobs=-1
)
lr_search.fit(X_train_scaled, y_train)
best_lr = lr_search.best_estimator_
pred = best_lr.predict(X_val_scaled)
proba = best_lr.predict_proba(X_val_scaled)[:, 1]
results.append(evaluate("Logistic Regression (tuned)", y_val, pred, proba, lr_search.best_params_))
joblib.dump(best_lr, f"{MODEL_DIR}/lr_tuned.joblib")

# ---------------- Random Forest (tuned) ----------------
rf_param_dist = {
    "n_estimators": [150, 250, 350],
    "max_depth": [6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=10, scoring="f1", cv=3, random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
pred = best_rf.predict(X_val)
proba = best_rf.predict_proba(X_val)[:, 1]
results.append(evaluate("Random Forest (tuned)", y_val, pred, proba, rf_search.best_params_))
joblib.dump(best_rf, f"{MODEL_DIR}/rf_tuned.joblib")

with open(f"{RESULTS_DIR}/tuned_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
