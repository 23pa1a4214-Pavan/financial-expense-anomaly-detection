Financial Expense Anomaly Detection for Corporate Auditing
Problem Statement

Corporate organizations process numerous expense transactions, making manual auditing time-consuming and potentially ineffective at identifying unusual expenses. This project develops a machine learning system to detect potentially high-risk expense transactions, prioritize audit reviews, and explain predictions using Explainable AI.
Objectives

    Analyze historical expense transactions.

    Establish statistical baselines for normal spending patterns.

    Identify unusual expenses and potential policy violations.

    Train and evaluate a supervised machine learning risk classifier.

    Interpret predictions using SHAP.

    Prioritize transactions for audit review.

    Generate an audit report.

    Deploy an interactive application using Streamlit.

Input Data

Transaction amount, department, expense category, transaction date, payment method, and available duplicate or policy-violation indicators.
Expected Outputs

    Expense risk classification model

    Transaction-level risk scores

    Statistical anomaly indicators

    SHAP explanations

    Audit-priority list

    Downloadable audit report

    Streamlit web application

Evaluation

Evaluate precision, recall, F1-score, PR-AUC, and false-positive rates using held-out data.
Limitations

An anomaly is not necessarily fraud. Model predictions are intended to support human auditors, not replace audit judgment. Synthetic labels, if used, must be clearly distinguished from verified audit outcomes.