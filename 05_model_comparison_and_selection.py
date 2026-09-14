"""
Step 3d / Step 4: Model Comparison & Final Selection + Evaluation
HR Analytics Employee Attrition & Performance
--------------------------------------------------------
IMPORTANT (data leakage / methodology discipline):
  - All hyperparameter tuning decisions above were made using ONLY
    train + validation data.
  - The TEST set is touched here for the FIRST and ONLY time, to produce
    the final, honest performance comparison. We do not go back and
    re-tune based on test results.

This script:
  1. Loads all available TUNED models (LR, RF, and XGBoost if you've
     copied xgboost_metrics.json / xgb_tuned.joblib back from your
     local run).
  2. Evaluates each on the TEST set.
  3. Builds a comparison table.
  4. Selects the final model using a documented rule (see SELECTION_RULE).
  5. Saves the final model + final metrics + confusion matrix.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

test = pd.read_csv(f"{DATA_DIR}/test.csv")
X_test, y_test = test.drop(columns=["Attrition"]), test["Attrition"]

scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
X_test_scaled = scaler.transform(X_test)

# Selection rule, decided in advance (do NOT change after seeing results):
#   Primary:   Recall on the attrition="Yes" class must be >= 0.70
#              (missing a real leaver is the costlier error for this business problem)
#   Tie-break: Among models clearing that bar, pick the highest F1 score
#              (so we don't win recall by flagging almost everyone)
MIN_RECALL_BAR = 0.70


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


candidates = []

# ---- Logistic Regression (tuned) ----
lr = joblib.load(f"{MODEL_DIR}/lr_tuned.joblib")
pred = lr.predict(X_test_scaled)
proba = lr.predict_proba(X_test_scaled)[:, 1]
candidates.append(("Logistic Regression (tuned)", lr, evaluate("Logistic Regression (tuned)", y_test, pred, proba), "sklearn_scaled"))

# ---- Random Forest (tuned) ----
rf = joblib.load(f"{MODEL_DIR}/rf_tuned.joblib")
pred = rf.predict(X_test)
proba = rf.predict_proba(X_test)[:, 1]
candidates.append(("Random Forest (tuned)", rf, evaluate("Random Forest (tuned)", y_test, pred, proba), "sklearn_raw"))

# ---- XGBoost (tuned) -- only if you've copied it back from the local run ----
xgb_path = f"{MODEL_DIR}/xgb_tuned.joblib"
if os.path.exists(xgb_path):
    xgb = joblib.load(xgb_path)
    pred = xgb.predict(X_test)
    proba = xgb.predict_proba(X_test)[:, 1]
    candidates.append(("XGBoost (tuned)", xgb, evaluate("XGBoost (tuned)", y_test, pred, proba), "sklearn_raw"))
else:
    print("NOTE: xgb_tuned.joblib not found -- run 04_train_xgboost_LOCAL.py "
          "elsewhere and copy models/xgb_tuned.joblib + results/xgboost_metrics.json "
          "into this project before final submission.\n")

comparison_table = [c[2] for c in candidates]
print("=== TEST SET COMPARISON (touched once) ===")
print(json.dumps(comparison_table, indent=2))

# ---- Apply selection rule ----
qualifying = [c for c in candidates if c[2]["recall"] >= MIN_RECALL_BAR]
pool = qualifying if qualifying else candidates  # fall back if nobody clears the bar
final_name, final_model, final_metrics, final_kind = max(pool, key=lambda c: c[2]["f1_score"])

print(f"\n=== FINAL MODEL SELECTED: {final_name} ===")
print(f"Selection rule: Recall >= {MIN_RECALL_BAR} required, then highest F1 wins.")
print(json.dumps(final_metrics, indent=2))

# Save final artifacts
joblib.dump(final_model, f"{MODEL_DIR}/final_model.joblib")
with open(f"{MODEL_DIR}/final_model_info.json", "w") as f:
    json.dump({"name": final_name, "feature_kind": final_kind}, f, indent=2)

with open(f"{RESULTS_DIR}/test_comparison.json", "w") as f:
    json.dump(comparison_table, f, indent=2)

with open(f"{RESULTS_DIR}/final_model_metrics.json", "w") as f:
    json.dump({"selected_model": final_name, "metrics": final_metrics,
                "selection_rule": f"recall>={MIN_RECALL_BAR} then max F1"}, f, indent=2)
