# Problem Definition: HR Attrition Risk & Retention Management

## Executive Summary
Employee attrition imposes significant financial costs, loss of institutional knowledge, and operational disruption. The objective of this project is to build an end-to-end Machine Learning pipeline and interactive HR decision-support dashboard to proactively identify high-risk employees, model potential retention intervention strategies, and quantify the net financial impact of retention efforts.

---

## Business Problem & Strategic Objectives

* **Core Challenge:** HR teams currently react to resignations after notice is given, leaving little room for intervention.
* **Target Outcome:** Identify flight-risk employees *before* they resign and equip HR business partners with data-driven counterfactual actions (e.g., compensation adjustments, workload reduction).
* **Primary Business Goal:** Maximize **Net Retention Cost Saved** across the enterprise while minimizing unneeded intervention spending on false positives.

---

## Machine Learning Problem Formulation

* **Task Type:** Binary Classification (Supervised Learning)
* **Target Variable:** `Attrition` 
  * `1` = Employee leaves the organization
  * `0` = Employee stays with the organization
* **Model Inputs:** Tabular workforce data including demographics, compensation, job role parameters, satisfaction surveys, and engineered indicators.
* **Data Leakage Mitigation:** The explicit proxy metric `Attrition_Risk_Score` is removed prior to model training and isolated into a baseline reference set.

---

## Success Metrics & Target KPIs

| KPI Category | KPI Metric Name | Definition / Formula | Target Threshold |
| :--- | :--- | :--- | :--- |
| **Business** | Net Retention Cost Saved | $(TP \times C_{\text{replace}} \times R_{\text{success}}) - ((TP + FP) \times C_{\text{intervene}})$ | $> \$0$ |
| **Machine Learning** | Flight Risk Recall | $\frac{TP}{TP + FN}$ | $\ge 0.75$ |
| **Machine Learning** | Model F1 Score | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | $\ge 0.60$ |
| **Data Quality** | Feature Completeness Rate | $1 - \left(\frac{\text{Missing Values}}{\text{Total Required Cells}}\right)$ | $\ge 0.95$ |
| **Engineering** | Counterfactual Latency | Average single-query what-if simulation execution time | $< 500\text{ ms}$ |

---

## Operating Assumptions & Baseline Economics

* **Average Replacement Cost:** Equivalent to **9 months** of an employee's base monthly salary ($C_{\text{replace}} = 9 \times \bar{\text{Monthly\_Income}}$).
* **Intervention Success Rate ($R_{\text{success}}$):** $30\%$ of correctly identified leavers are successfully retained following a targeted intervention strategy.
* **Intervention Cost ($C_{\text{intervene}}$):** $\$2,000$ per targeted employee (covers retention bonuses, salary adjustments, or professional development programs).
* **Model Selection Rule:** Primary threshold requires **Recall $\ge 0.70$** on test evaluation, using **F1 Score** as the primary tie-breaker.

---

## Scope & Constraints

* **In-Scope:** Feature engineering, baseline and ensemble model experimentation, automated threshold selection, financial KPI modeling, and Streamlit user interface development.
* **Out-of-Scope:** Real-time streaming database connectivity, external job-market scraping, and unstructured sentiment analysis from performance appraisal text.