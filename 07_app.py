"""
Step 6: Data Product -- Streamlit Dashboard
HR Analytics Employee Attrition & Performance
--------------------------------------------------------
Run with:
    streamlit run 07_app.py

Requires (in the same folder):
    models/final_model.joblib
    models/final_model_info.json
    models/scaler.joblib
    data/train.csv   (used to get the feature schema / defaults)
    results/kpi_report.json (optional, for the KPI tab)
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="HR Attrition Risk Dashboard", layout="wide")

MODEL_DIR = "models"
DATA_DIR = "data"
RESULTS_DIR = "results"


def risk_gauge(prob: float, title: str = "Attrition Risk"):
    """A color-zoned gauge (green/yellow/red) for a single probability, 0-1."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1F4E79"},
            "steps": [
                {"range": [0, 40], "color": "#C6E0B4"},   # low - green
                {"range": [40, 70], "color": "#FFE699"},  # medium - yellow
                {"range": [70, 100], "color": "#F8CBAD"}, # high - red
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODEL_DIR}/final_model.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    with open(f"{MODEL_DIR}/final_model_info.json") as f:
        info = json.load(f)
    train_schema = pd.read_csv(f"{DATA_DIR}/train.csv").drop(columns=["Attrition"])
    return model, scaler, info, train_schema


model, scaler, model_info, schema = load_artifacts()
FEATURE_COLS = schema.columns.tolist()


def score(df: pd.DataFrame) -> np.ndarray:
    """Apply the same preprocessing used at train time and return P(attrition)."""
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    X = scaler.transform(df) if model_info["feature_kind"] == "sklearn_scaled" else df
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# The uploaded CSV might be in either of two valid shapes:
#   1. RAW    -- the original dataset format (Department="Sales", Overtime="Yes",
#                no engineered features, not one-hot encoded).
#   2. ENCODED -- already matches data/train.csv / test.csv exactly (this is
#                what 01_data_engineering.py produces).
# Both are legitimate "employee datasets"; only something unrelated (e.g. an
# Instagram analytics export) should be rejected.
# ---------------------------------------------------------------------------
RAW_REQUIRED_COLUMNS = [
    "Age", "Gender", "Marital_Status", "Education_Level", "Department", "Job_Role",
    "Employment_Type", "Years_At_Company", "Years_In_Current_Role", "Monthly_Income",
    "Salary_Hike_Percent", "Job_Level", "Performance_Rating", "Job_Satisfaction",
    "Work_Life_Balance", "Environment_Satisfaction", "Relationship_Satisfaction",
    "Training_Hours_Last_Year", "Overtime", "Remote_Work", "Business_Travel",
    "Distance_From_Home", "Commute_Time_Minutes", "Number_Of_Projects", "Team_Size",
    "Manager_Rating", "Absence_Days", "Promotion_Last_5_Years", "Stock_Option",
]


def detect_format(df: pd.DataFrame) -> str:
    """Returns 'raw', 'encoded', or 'unknown' based on column-name overlap."""
    raw_overlap = len(set(df.columns) & set(RAW_REQUIRED_COLUMNS)) / len(RAW_REQUIRED_COLUMNS)
    encoded_overlap = len(set(df.columns) & set(FEATURE_COLS)) / len(FEATURE_COLS)
    if raw_overlap >= 0.9:
        return "raw"
    if encoded_overlap >= 0.5:
        return "encoded"
    return "unknown"


def preprocess_raw_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors 01_data_engineering.py's feature engineering + encoding, applied
    to a batch of raw-format rows. Reindexing to FEATURE_COLS at the end
    correctly zeroes out any one-hot category that was the training-time
    reference category (dropped via drop_first) or wasn't seen in this batch."""
    df = df.copy()
    df = df.drop(columns=[c for c in ["Employee_ID", "Attrition_Risk_Score", "Attrition"] if c in df.columns])

    df["Tenure_Stability_Ratio"] = df["Years_In_Current_Role"] / (df["Years_At_Company"] + 1)
    df["Income_per_JobLevel"] = df["Monthly_Income"] / df["Job_Level"]
    satisfaction_cols = ["Job_Satisfaction", "Work_Life_Balance", "Environment_Satisfaction", "Relationship_Satisfaction"]
    df["Satisfaction_Composite"] = df[satisfaction_cols].mean(axis=1)
    df["Commute_Burden"] = df["Distance_From_Home"] + (df["Commute_Time_Minutes"] / 10.0)
    df["Promotion_Stagnation_Flag"] = ((df["Promotion_Last_5_Years"] == "No") & (df["Years_At_Company"] > 3)).astype(int)
    df["High_Overtime_Low_WLB"] = ((df["Overtime"] == "Yes") & (df["Work_Life_Balance"] <= 2)).astype(int)

    binary_map = {"Yes": 1, "No": 0}
    for col in ["Overtime", "Remote_Work", "Promotion_Last_5_Years", "Stock_Option"]:
        df[col] = df[col].map(binary_map)

    education_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    df["Education_Level"] = df["Education_Level"].map(education_order)

    nominal_cols = ["Gender", "Marital_Status", "Department", "Job_Role", "Employment_Type", "Business_Travel"]
    df = pd.get_dummies(df, columns=nominal_cols)  # no drop_first: reindex below handles the reference category

    return df.reindex(columns=FEATURE_COLS, fill_value=0)


# ---------------------------------------------------------------------------
# Helpers for the "Individual Prediction" tab: the training data was already
# one-hot encoded (e.g. "Department_Sales", "Job_Role_SEO Specialist"), so we
# recover the original category options by reading the dummy column names.
# Any category NOT listed (the one dropped by drop_first=True during training)
# is offered as "Other / Unlisted" -- selecting it just leaves all of that
# field's dummy columns at 0, which is exactly how the reference category
# was represented at training time.
# ---------------------------------------------------------------------------
NOMINAL_PREFIXES = [
    "Gender", "Marital_Status", "Department",
    "Job_Role", "Employment_Type", "Business_Travel",
]


def get_categories(prefix: str):
    cats = [c[len(prefix) + 1:] for c in FEATURE_COLS if c.startswith(prefix + "_")]
    return cats + ["Other / Unlisted"]


def build_input_row(values: dict) -> pd.DataFrame:
    """Turn a dict of human-entered raw values into one encoded row matching
    FEATURE_COLS, applying the same feature engineering as 01_data_engineering.py."""
    row = pd.DataFrame([{c: 0 for c in FEATURE_COLS}])

    # Direct numeric / already-binary fields
    direct_fields = [
        "Age", "Years_At_Company", "Years_In_Current_Role", "Monthly_Income",
        "Salary_Hike_Percent", "Job_Level", "Performance_Rating", "Job_Satisfaction",
        "Work_Life_Balance", "Environment_Satisfaction", "Relationship_Satisfaction",
        "Training_Hours_Last_Year", "Distance_From_Home", "Commute_Time_Minutes",
        "Number_Of_Projects", "Team_Size", "Manager_Rating", "Absence_Days",
    ]
    for f in direct_fields:
        if f in row.columns:
            row[f] = values[f]

    # Binary Yes/No fields
    for f in ["Overtime", "Remote_Work", "Promotion_Last_5_Years", "Stock_Option"]:
        if f in row.columns:
            row[f] = 1 if values[f] == "Yes" else 0

    # Ordinal education
    education_order = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
    if "Education_Level" in row.columns:
        row["Education_Level"] = education_order[values["Education_Level"]]

    # One-hot nominal fields
    for prefix in NOMINAL_PREFIXES:
        selected = values.get(prefix)
        col_name = f"{prefix}_{selected}"
        if col_name in row.columns:
            row[col_name] = 1
        # if "Other / Unlisted" or a dropped reference category, leave all-0

    # Engineered features (must mirror 01_data_engineering.py exactly)
    if "Tenure_Stability_Ratio" in row.columns:
        row["Tenure_Stability_Ratio"] = values["Years_In_Current_Role"] / (values["Years_At_Company"] + 1)
    if "Income_per_JobLevel" in row.columns:
        row["Income_per_JobLevel"] = values["Monthly_Income"] / values["Job_Level"]
    if "Satisfaction_Composite" in row.columns:
        row["Satisfaction_Composite"] = np.mean([
            values["Job_Satisfaction"], values["Work_Life_Balance"],
            values["Environment_Satisfaction"], values["Relationship_Satisfaction"],
        ])
    if "Commute_Burden" in row.columns:
        row["Commute_Burden"] = values["Distance_From_Home"] + (values["Commute_Time_Minutes"] / 10.0)
    if "Promotion_Stagnation_Flag" in row.columns:
        row["Promotion_Stagnation_Flag"] = int(
            values["Promotion_Last_5_Years"] == "No" and values["Years_At_Company"] > 3
        )
    if "High_Overtime_Low_WLB" in row.columns:
        row["High_Overtime_Low_WLB"] = int(
            values["Overtime"] == "Yes" and values["Work_Life_Balance"] <= 2
        )

    return row


st.title("🧑‍💼 HR Attrition Risk Dashboard")
st.caption(f"Model in production: **{model_info['name']}**")

tab1, tab2, tab3, tab4 = st.tabs([
    "----[ Flight Risk List ]----", "----[ Counterfactual Simulator ]----",
    "----[ KPI Report ]----", "----[ Individual Prediction ]----",
])

# =========================================================
# TAB 1 -- Flight Risk List (batch scoring)
# =========================================================
with tab1:
    st.subheader("Score a batch of employees")
    uploaded = st.file_uploader(
        "Upload a CSV with the same columns as the training data "
        "(Employee_ID and Attrition columns are optional/ignored).",
        type=["csv"],
    )

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        feats = raw.drop(columns=[c for c in ["Employee_ID", "Attrition"] if c in raw.columns])

        fmt = detect_format(feats)

        if fmt == "unknown":
            raw_matches = len(set(feats.columns) & set(RAW_REQUIRED_COLUMNS))
            st.error(
                f"⚠️ This file doesn't look like an employee attrition dataset "
                f"(only {raw_matches} of {len(RAW_REQUIRED_COLUMNS)} expected raw columns matched, "
                f"and it doesn't match the encoded training format either). "
                "Please upload either the original-style employee CSV (columns like Age, Department, "
                "Job_Role, Monthly_Income, Overtime, Job_Satisfaction, ...) or a pre-processed file "
                "matching data/test.csv. "
                "You can use data/test.csv from the project as a sample."
            )
            st.stop()

        if fmt == "raw":
            st.caption("Detected raw-format employee data — applying feature engineering and encoding automatically.")
            feats_encoded = preprocess_raw_batch(feats)
        else:
            feats_encoded = feats

        id_col = raw["Employee_ID"] if "Employee_ID" in raw.columns else pd.Series(range(len(raw)))
        probs = score(feats_encoded)

        out = pd.DataFrame({
            "Employee_ID": id_col,
            "Attrition_Probability": probs,
        }).sort_values("Attrition_Probability", ascending=False)
        out["Risk_Tier"] = pd.cut(
            out["Attrition_Probability"], bins=[-0.01, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Employees scored", len(out))
        c2.metric("High risk (>70%)", int((out["Risk_Tier"] == "High").sum()))
        c3.metric("Avg. predicted risk", f"{out['Attrition_Probability'].mean():.1%}")

        chart1, chart2 = st.columns(2)
        with chart1:
            tier_counts = out["Risk_Tier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig_bar = px.bar(
                x=tier_counts.index, y=tier_counts.values,
                color=tier_counts.index,
                color_discrete_map={"Low": "#70AD47", "Medium": "#FFC000", "High": "#C00000"},
                labels={"x": "Risk Tier", "y": "Number of Employees"},
                title="Headcount by Risk Tier",
            )
            fig_bar.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig_bar, use_container_width=True)
        with chart2:
            fig_hist = px.histogram(
                out, x="Attrition_Probability", nbins=25,
                title="Distribution of Predicted Attrition Probability",
                labels={"Attrition_Probability": "Predicted Probability"},
            )
            fig_hist.update_traces(marker_color="#1F4E79")
            fig_hist.update_layout(height=320)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(out, use_container_width=True, height=420)
        st.download_button(
            "Download flight-risk list as CSV",
            out.to_csv(index=False).encode(),
            "flight_risk_list.csv",
            "text/csv",
        )
    else:
        st.info("Upload a CSV to see ranked flight-risk predictions here. "
                 "You can use data/test.csv from the project as a sample.")

# =========================================================
# TAB 2 -- Counterfactual Simulator (single employee, what-if)
# =========================================================
with tab2:
    st.subheader("What-if simulator for a single employee")
    st.caption("Adjust the levers HR can actually pull, and see how predicted "
               "attrition risk changes in real time.")

    default_row = schema.iloc[[0]].copy()

    colA, colB = st.columns(2)
    with colA:
        monthly_income = st.slider("Monthly Income", 1500, 400000,
                                    int(default_row["Monthly_Income"].values[0]), step=1000)
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        wlb = st.slider("Work-Life Balance (1=Poor, 10=Excellent)", 1, 10,
                         int(default_row["Work_Life_Balance"].values[0]))
    with colB:
        salary_hike = st.slider("Salary Hike % (last cycle)", 0.0, 30.0,
                                 float(default_row["Salary_Hike_Percent"].values[0]))
        promotion = st.selectbox("Promoted in last 5 years?", ["No", "Yes"])
        job_level = st.slider("Job Level", 1, 5, int(default_row["Job_Level"].values[0]))

    scenario = default_row.copy()
    scenario["Monthly_Income"] = monthly_income
    scenario["Overtime"] = 1 if overtime == "Yes" else 0
    scenario["Work_Life_Balance"] = wlb
    scenario["Salary_Hike_Percent"] = salary_hike
    scenario["Promotion_Last_5_Years"] = 1 if promotion == "Yes" else 0
    scenario["Job_Level"] = job_level

    # IMPORTANT: the engineered features below were computed ONCE at training
    # time from each employee's own raw values. Changing a raw slider above
    # does NOT automatically update them -- without this recalculation step,
    # the model would still see the *original* employee's stale engineered
    # values (e.g. an old High_Overtime_Low_WLB flag of 0) even after you set
    # Overtime=Yes and Work-Life Balance=1 here, badly understating the real
    # impact of the scenario. We recompute every engineered feature that
    # depends on a slider above, using the same formulas as
    # 01_data_engineering.py, so the model sees a fully consistent scenario.
    if "Income_per_JobLevel" in scenario.columns:
        scenario["Income_per_JobLevel"] = monthly_income / job_level
    if "High_Overtime_Low_WLB" in scenario.columns:
        scenario["High_Overtime_Low_WLB"] = int(overtime == "Yes" and wlb <= 2)
    if "Promotion_Stagnation_Flag" in scenario.columns:
        years_at_company = default_row["Years_At_Company"].values[0]
        scenario["Promotion_Stagnation_Flag"] = int(promotion == "No" and years_at_company > 3)
    if "Satisfaction_Composite" in scenario.columns:
        other_satisfaction = [
            default_row["Job_Satisfaction"].values[0],
            default_row["Environment_Satisfaction"].values[0],
            default_row["Relationship_Satisfaction"].values[0],
        ]
        scenario["Satisfaction_Composite"] = np.mean([wlb] + other_satisfaction)

    start = time.perf_counter()
    prob = score(scenario)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    baseline_prob = score(default_row)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline risk", f"{baseline_prob:.1%}")
    c2.metric("Scenario risk", f"{prob:.1%}", delta=f"{(prob - baseline_prob):.1%}")
    c3.metric("Simulation latency", f"{latency_ms:.1f} ms")

    gauge_col, compare_col = st.columns(2)
    with gauge_col:
        st.plotly_chart(risk_gauge(prob, "Scenario Risk"), use_container_width=True)
    with compare_col:
        fig_compare = px.bar(
            x=["Baseline", "Scenario"], y=[baseline_prob * 100, prob * 100],
            color=["Baseline", "Scenario"],
            color_discrete_map={"Baseline": "#8FAADC", "Scenario": "#1F4E79"},
            labels={"x": "", "y": "Attrition Risk (%)"},
            title="Baseline vs. Scenario Risk",
        )
        fig_compare.update_layout(showlegend=False, height=280, yaxis_range=[0, 100])
        st.plotly_chart(fig_compare, use_container_width=True)

    if prob >= 0.7:
        st.error("High flight risk under this scenario.")
    elif prob >= 0.4:
        st.warning("Medium flight risk under this scenario.")
    else:
        st.success("Low flight risk under this scenario.")

# =========================================================
# TAB 3 -- KPI Report
# =========================================================
with tab3:
    st.subheader("Project KPIs")

    # ---- Model comparison chart (Recall & F1 across all tuned models) ----
    try:
        with open(f"{RESULTS_DIR}/test_comparison.json") as f:
            comparison = json.load(f)
        comp_df = pd.DataFrame(comparison)
        melted = comp_df.melt(
            id_vars="model", value_vars=["recall", "f1_score", "roc_auc"],
            var_name="Metric", value_name="Score"
        )
        fig_models = px.bar(
            melted, x="model", y="Score", color="Metric", barmode="group",
            title="Model Comparison on Held-Out Test Set",
            labels={"model": "", "Score": ""},
        )
        fig_models.update_layout(height=380, yaxis_range=[0, 1])
        st.plotly_chart(fig_models, use_container_width=True)
    except FileNotFoundError:
        st.info("results/test_comparison.json not found -- run 05_model_comparison_and_selection.py first.")

    st.divider()

    try:
        with open(f"{RESULTS_DIR}/kpi_report.json") as f:
            kpis = json.load(f)

        # ---- Target vs Actual bars for the numeric-target KPIs ----
        numeric_kpis = [k for k in kpis if isinstance(k["target"], (int, float))]
        if numeric_kpis:
            gauge_cols = st.columns(len(numeric_kpis))
            for col, kpi in zip(gauge_cols, numeric_kpis):
                with col:
                    fig_kpi = go.Figure()
                    fig_kpi.add_trace(go.Bar(
                        x=["Target", "Actual"],
                        y=[kpi["target"], kpi["actual_result"]],
                        marker_color=["#8FAADC", "#1F4E79" if kpi["actual_result"] >= kpi["target"] else "#C00000"],
                    ))
                    fig_kpi.update_layout(
                        title=kpi["name"], height=280,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_kpi, use_container_width=True)

        # ---- Full KPI detail (formula, assumptions, interpretation) ----
        for kpi in kpis:
            with st.expander(f"**{kpi['name']}** — target: {kpi['target']} | actual: {kpi['actual_result']}"):
                st.write(f"**Formula:** {kpi['formula']}")
                if "assumptions" in kpi:
                    st.write("**Assumptions:**", kpi["assumptions"])
                st.write(f"**Interpretation:** {kpi['interpretation']}")
    except FileNotFoundError:
        st.warning("results/kpi_report.json not found -- run 06_kpi_report.py first.")

# =========================================================
# TAB 4 -- Individual Prediction (manual single-employee entry)
# =========================================================
with tab4:
    st.subheader("Enter one employee's details to get their flight risk")

    with st.form("individual_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 70, 30)
            gender = st.selectbox("Gender", get_categories("Gender"))
            marital_status = st.selectbox("Marital Status", get_categories("Marital_Status"))
            education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
            department = st.selectbox("Department", get_categories("Department"))
            job_role = st.selectbox("Job Role", get_categories("Job_Role"))
            employment_type = st.selectbox("Employment Type", get_categories("Employment_Type"))
            business_travel = st.selectbox("Business Travel", get_categories("Business_Travel"))
            job_level = st.slider("Job Level", 1, 5, 1)

        with c2:
            years_at_company = st.number_input("Years At Company", 0, 40, 3)
            years_in_role = st.number_input("Years In Current Role", 0, 40, 1)
            monthly_income = st.number_input("Monthly Income", 1000, 400000, 8000, step=1000)
            salary_hike = st.number_input("Salary Hike % (last cycle)", 0.0, 30.0, 12.0)
            performance_rating = st.slider("Performance Rating", 1, 10, 3)
            manager_rating = st.slider("Manager Rating", 1, 10, 3)
            number_of_projects = st.number_input("Number Of Projects", 0, 30, 5)
            team_size = st.number_input("Team Size", 1, 50, 8)

        with c3:
            job_satisfaction = st.slider("Job Satisfaction", 1, 10, 3)
            work_life_balance = st.slider("Work-Life Balance", 1, 10, 3)
            environment_satisfaction = st.slider("Environment Satisfaction", 1, 10, 3)
            relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 10, 3)
            training_hours = st.number_input("Training Hours Last Year", 0, 200, 30)
            distance_from_home = st.number_input("Distance From Home (km)", 0, 100, 10)
            commute_time = st.number_input("Commute Time (minutes)", 0, 180, 30)
            absence_days = st.number_input("Absence Days", 0, 60, 5)

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            overtime = st.selectbox("Overtime", ["No", "Yes"])
        with c5:
            remote_work = st.selectbox("Remote Work", ["No", "Yes"])
        with c6:
            promotion = st.selectbox("Promoted in last 5 years?", ["No", "Yes"])
        with c7:
            stock_option = st.selectbox("Has Stock Option?", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Flight Risk")

    if submitted:
        values = dict(
            Age=age, Gender=gender, Marital_Status=marital_status,
            Education_Level=education_level, Department=department, Job_Role=job_role,
            Employment_Type=employment_type, Business_Travel=business_travel,
            Job_Level=job_level, Years_At_Company=years_at_company,
            Years_In_Current_Role=years_in_role, Monthly_Income=monthly_income,
            Salary_Hike_Percent=salary_hike, Performance_Rating=performance_rating,
            Manager_Rating=manager_rating, Number_Of_Projects=number_of_projects,
            Team_Size=team_size, Job_Satisfaction=job_satisfaction,
            Work_Life_Balance=work_life_balance,
            Environment_Satisfaction=environment_satisfaction,
            Relationship_Satisfaction=relationship_satisfaction,
            Training_Hours_Last_Year=training_hours,
            Distance_From_Home=distance_from_home, Commute_Time_Minutes=commute_time,
            Absence_Days=absence_days, Overtime=overtime, Remote_Work=remote_work,
            Promotion_Last_5_Years=promotion, Stock_Option=stock_option,
        )

        start = time.perf_counter()
        input_row = build_input_row(values)
        prob = score(input_row)[0]
        latency_ms = (time.perf_counter() - start) * 1000

        st.divider()
        r1, r2 = st.columns(2)
        r1.metric("Predicted Attrition Probability", f"{prob:.1%}")
        r2.metric("Prediction latency", f"{latency_ms:.1f} ms")

        st.plotly_chart(risk_gauge(prob, "Employee Flight Risk"), use_container_width=True)

        if prob >= 0.7:
            st.error("🔴 High flight risk — recommend a retention conversation soon.")
        elif prob >= 0.4:
            st.warning("🟡 Medium flight risk — worth monitoring.")
        else:
            st.success("🟢 Low flight risk.")