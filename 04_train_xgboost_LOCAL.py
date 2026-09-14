"""
Step 3c: XGBoost (RUN THIS LOCALLY / ON COLAB)
HR Analytics Employee Attrition & Performance
--------------------------------------------------------
This mirrors 02_train_baseline_models.py + 03_hyperparameter_tuning.py
but for XGBoost. Run it in an environment with internet access, e.g.:

    pip install xgboost scikit-learn pandas joblib

It expects the SAME train.csv / val.csv produced by 01_data_engineering.py
to sit in ./data/ next to this script (copy the data/ folder over).

Produces:
  - models/xgb_baseline.joblib
  - models/xgb_tuned.joblib
  - results/xgboost_metrics.json   <-- copy this back to merge into the
                                        model comparison / final report
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

train = pd.read_csv(f"{DATA_DIR}/train.csv")
val = pd.read_csv(f"{DATA_DIR}/val.csv")

X_train, y_train = train.drop(columns=["Attrition"]), train["Attrition"]
X_val, y_val = val.drop(columns=["Attrition"]), val["Attrition"]

# Ratio of negative/positive class, used to rebalance training
# (XGBoost's analogue of class_weight='balanced')
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (neg/pos ratio) = {scale_pos_weight:.3f}")


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

# ---------------- XGBoost (baseline, default hyperparameters) ----------------
xgb_baseline = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb_baseline.fit(X_train, y_train)
pred = xgb_baseline.predict(X_val)
proba = xgb_baseline.predict_proba(X_val)[:, 1]
results.append(evaluate("XGBoost (baseline)", y_val, pred, proba))
joblib.dump(xgb_baseline, f"{MODEL_DIR}/xgb_baseline.joblib")

# ---------------- XGBoost (tuned) ----------------
# scale_pos_weight is INCLUDED as a tunable value (not fixed) so the search
# can also try the "unbalanced" default (1.0) and pick whichever wins on F1.
param_dist = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "scale_pos_weight": [1.0, scale_pos_weight],
}

search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    param_distributions=param_dist,
    n_iter=30,
    scoring="f1",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)
best_xgb = search.best_estimator_
pred = best_xgb.predict(X_val)
proba = best_xgb.predict_proba(X_val)[:, 1]
results.append(evaluate("XGBoost (tuned)", y_val, pred, proba, search.best_params_))
joblib.dump(best_xgb, f"{MODEL_DIR}/xgb_tuned.joblib")

with open(f"{RESULTS_DIR}/xgboost_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\n--> Copy results/xgboost_metrics.json and models/xgb_tuned.joblib")
print("    back into the main project folder before running 05_final_evaluation.py")
