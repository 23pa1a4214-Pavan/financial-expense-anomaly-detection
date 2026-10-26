
import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/expense_risk_model.joblib"
DATA_PATH = "data/engineered_expense_features.csv"
THRESHOLD_PATH = "reports/risk_threshold.json"

OUTPUT_PATH = "reports/audit_priority_register.csv"
SUMMARY_PATH = "reports/audit_priority_summary.json"

os.makedirs("reports", exist_ok=True)

# Fields excluded from model inputs during training
EXCLUDED_COLUMNS = [
    "transaction_id",
    "employee_id",
    "transaction_date",
    "high_risk",
    "synthetic_label_reason",
    "statistical_anomaly",
    "is_duplicate",
    "policy_violation",
    "amount_to_median_ratio",
    "robust_z_score",
    "category_median",
    "category_mad",
    "category_percentile",
    "employee_avg_amount",
    "amount_to_employee_avg_ratio",
    "audit_label",
]

# 1. Load model, threshold, and transaction data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_info = json.load(f)

threshold = float(threshold_info["threshold"])

print("=" * 65)
print("CORPORATE AUDIT PRIORITIZATION")
print("=" * 65)

# 2. Prepare exactly the model's input columns
X = df.drop(columns=EXCLUDED_COLUMNS, errors="ignore")

expected_features = list(
    model.named_steps["preprocessor"].feature_names_in_
)

missing_features = set(expected_features) - set(X.columns)

if missing_features:
    raise ValueError(
        f"Missing model input features: {sorted(missing_features)}"
    )

X = X[expected_features]

# 3. Predict high-risk probabilities
df["risk_probability"] = model.predict_proba(X)[:, 1]

df["model_high_risk"] = (
    df["risk_probability"] >= threshold
).astype(int)

# 4. Calculate financial exposure for review
# This is the claimed amount associated with a flagged transaction,
# NOT a confirmed loss or recoverable amount.
df["potential_exposure_inr"] = np.where(
    df["model_high_risk"] == 1,
    df["amount"],
    0.0,
)

# 5. Build a transparent operational priority score
# Risk probability: 70% of score
# Relative amount percentile: 20%
# Additional warning indicators: 10%
df["amount_percentile"] = (
    df["amount"].rank(pct=True) * 100
)

duplicate_signal = (
    df["is_duplicate"].fillna(0).astype(float)
    if "is_duplicate" in df.columns
    else 0.0
)

policy_signal = (
    df["policy_violation"].fillna(0).astype(float)
    if "policy_violation" in df.columns
    else 0.0
)

statistical_signal = (
    df["statistical_anomaly"].fillna(0).astype(float)
    if "statistical_anomaly" in df.columns
    else 0.0
)

df["additional_warning_score"] = (
    0.4 * duplicate_signal
    + 0.4 * policy_signal
    + 0.2 * statistical_signal
)

df["priority_score"] = (
    70 * df["risk_probability"]
    + 0.20 * df["amount_percentile"]
    + 10 * df["additional_warning_score"]
)

# 6. Assign priority levels
def assign_priority(row):
    if row["model_high_risk"] == 1:
        if (
            row["risk_probability"] >= 0.85
            and (
                row["amount_percentile"] >= 95
                or row["additional_warning_score"] >= 0.4
            )
        ):
            return "P1 - Urgent"

        return "P2 - High"

    # Keep lower-risk unusual transactions visible for monitoring.
    if (
        row["amount_percentile"] >= 95
        or row["additional_warning_score"] >= 0.4
    ):
        return "P3 - Routine"

    return "Monitor"

df["audit_priority"] = df.apply(
    assign_priority, axis=1
)

# 7. Sort the audit register
priority_order = {
    "P1 - Urgent": 0,
    "P2 - High": 1,
    "P3 - Routine": 2,
    "Monitor": 3,
}

df["priority_order"] = df["audit_priority"].map(priority_order)

df = df.sort_values(
    ["priority_order", "priority_score", "amount"],
    ascending=[True, False, False],
).reset_index(drop=True)

df["audit_rank"] = np.arange(1, len(df) + 1)

# 8. Save audit register
report_columns = [
    "audit_rank",
    "transaction_id",
    "employee_id",
    "department",
    "expense_category",
    "amount",
    "transaction_date",
    "risk_probability",
    "model_high_risk",
    "audit_priority",
    "priority_score",
    "potential_exposure_inr",
    "is_duplicate",
    "policy_violation",
    "statistical_anomaly",
]

report_columns = [
    col for col in report_columns if col in df.columns
]

df[report_columns].to_csv(
    OUTPUT_PATH, index=False
)

# 9. Generate summary statistics
priority_counts = (
    df["audit_priority"]
    .value_counts()
    .to_dict()
)

flagged = df[df["model_high_risk"] == 1]

summary = {
    "total_transactions": int(len(df)),
    "selected_threshold": threshold,
    "model_flagged_transactions": int(len(flagged)),
    "priority_counts": {
        key: int(value)
        for key, value in priority_counts.items()
    },
    "flagged_claim_amount_inr": round(
        float(flagged["amount"].sum()), 2
    ),
    "warning": (
        "Flagged claim amounts are potential exposure for review, "
        "not confirmed financial losses. Predictions use synthetic "
        "labels and require human audit verification."
    ),
}

with open(
    SUMMARY_PATH, "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print(f"Transactions processed: {len(df)}")
print(f"Model threshold: {threshold:.2f}")
print(f"Transactions flagged: {len(flagged)}")

print("\nAUDIT PRIORITY COUNTS")
print(df["audit_priority"].value_counts().to_string())

print("\nTOP 15 TRANSACTIONS FOR REVIEW")
print(
    df[report_columns]
    .head(15)
    .round(3)
    .to_string(index=False)
)

print("\nPotential exposure among flagged claims: "
      f"Rs. {flagged['amount'].sum():,.2f}")

print("\nSaved reports:")
print(OUTPUT_PATH)
print(SUMMARY_PATH)
