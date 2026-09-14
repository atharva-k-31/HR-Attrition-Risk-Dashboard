"""
Step 2: Data Engineering
HR Analytics Employee Attrition & Performance
--------------------------------------------------------
- Loads raw data
- Drops leakage / identifier columns
- Engineers new features
- Encodes categoricals
- Produces stratified train / val / test splits
- Saves everything + a data quality report
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

RAW_PATH = "data/employee_attrition_dataset.csv"  # place your raw CSV here if re-running this step
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
n_raw = len(df)

# ----------------------------------------------------------------
# 2. Data quality checks (Feature Completeness Rate KPI lives here)
# ----------------------------------------------------------------
required_cols = [c for c in df.columns if c not in ["Employee_ID", "Attrition_Risk_Score"]]
completeness = df[required_cols].notna().mean().mean()  # avg non-null rate across required fields
n_missing_total = df[required_cols].isna().sum().sum()
n_duplicates = df.duplicated().sum()

data_quality_report = {
    "n_rows_raw": int(n_raw),
    "n_columns_raw": int(df.shape[1]),
    "n_missing_values": int(n_missing_total),
    "n_duplicate_rows": int(n_duplicates),
    "feature_completeness_rate": round(float(completeness), 4),
}

# ----------------------------------------------------------------
# 3. Drop leakage / identifier columns
# ----------------------------------------------------------------
# Attrition_Risk_Score correlates 0.60 with target -> leakage proxy, dropped from
#   the modeling set (kept separately only for a leakage-demo comparison).
leaked_score = df[["Employee_ID", "Attrition_Risk_Score", "Attrition"]].copy()
leaked_score.to_csv(f"{OUT_DIR}/leakage_demo_reference.csv", index=False)

df = df.drop(columns=["Employee_ID", "Attrition_Risk_Score"])

# ----------------------------------------------------------------
# 4. Feature engineering
# ----------------------------------------------------------------
df["Tenure_Stability_Ratio"] = df["Years_In_Current_Role"] / (df["Years_At_Company"] + 1)

df["Income_per_JobLevel"] = df["Monthly_Income"] / df["Job_Level"]

satisfaction_cols = [
    "Job_Satisfaction", "Work_Life_Balance",
    "Environment_Satisfaction", "Relationship_Satisfaction"
]
df["Satisfaction_Composite"] = df[satisfaction_cols].mean(axis=1)

df["Commute_Burden"] = df["Distance_From_Home"] + (df["Commute_Time_Minutes"] / 10.0)

df["Promotion_Stagnation_Flag"] = (
    (df["Promotion_Last_5_Years"] == "No") & (df["Years_At_Company"] > 3)
).astype(int)

df["High_Overtime_Low_WLB"] = (
    (df["Overtime"] == "Yes") & (df["Work_Life_Balance"] <= 2)
).astype(int)

# ----------------------------------------------------------------
# 5. Encoding
# ----------------------------------------------------------------
binary_map = {"Yes": 1, "No": 0}
for col in ["Overtime", "Remote_Work", "Promotion_Last_5_Years", "Stock_Option"]:
    df[col] = df[col].map(binary_map)

education_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
df["Education_Level"] = df["Education_Level"].map(education_order)

nominal_cols = [
    "Gender", "Marital_Status", "Department",
    "Job_Role", "Employment_Type", "Business_Travel"
]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Target
df["Attrition"] = df["Attrition"].map(binary_map)

# ----------------------------------------------------------------
# 6. Train / Validation / Test split (stratified, 70/15/15)
# ----------------------------------------------------------------
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

for name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    out = Xs.copy()
    out["Attrition"] = ys
    out.to_csv(f"{OUT_DIR}/{name}.csv", index=False)

data_quality_report.update({
    "n_rows_after_cleaning": int(len(df)),
    "n_features_final": int(X.shape[1]),
    "train_size": int(len(X_train)),
    "val_size": int(len(X_val)),
    "test_size": int(len(X_test)),
    "train_positive_rate": round(float(y_train.mean()), 4),
    "val_positive_rate": round(float(y_val.mean()), 4),
    "test_positive_rate": round(float(y_test.mean()), 4),
})

with open(f"{OUT_DIR}/data_quality_report.json", "w") as f:
    json.dump(data_quality_report, f, indent=2)

print(json.dumps(data_quality_report, indent=2))
print("\nFinal feature columns:")
print(list(X.columns))
