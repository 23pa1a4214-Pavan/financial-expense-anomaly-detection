
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)

MODEL_PATH = "models/expense_risk_model.joblib"
VALIDATION_PATH = "data/splits/validation.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

TARGET = "audit_label"
TARGET_PRECISION = 0.80

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
]

# 1. Load the trained model and validation data
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train_model.py first."
    )

if not os.path.exists(VALIDATION_PATH):
    raise FileNotFoundError(
        f"Validation data not found: {VALIDATION_PATH}"
    )

model = joblib.load(MODEL_PATH)
df = pd.read_csv(VALIDATION_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Missing target column: {TARGET}")

y_true = df[TARGET].astype(int)

X_val = df.drop(columns=[TARGET])
X_val = X_val.drop(
    columns=EXCLUDED_COLUMNS,
    errors="ignore"
)

# 2. Predict risk probabilities
probabilities = model.predict_proba(X_val)[:, 1]

# 3. Evaluate several possible thresholds
thresholds = np.arange(0.10, 0.96, 0.05)
results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "flagged_transactions": int(tp + fp),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    f"{REPORT_DIR}/threshold_comparison.csv",
    index=False,
)

# 4. Select a threshold based on the precision target.
# Among thresholds meeting the target, choose the one
# with the highest recall.
eligible = results_df[
    (results_df["precision"] >= TARGET_PRECISION)
    & (results_df["flagged_transactions"] > 0)
]

if not eligible.empty:
    selected = eligible.sort_values(
        ["recall", "precision", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    selection_reason = (
        f"Highest recall among tested thresholds "
        f"meeting precision >= {TARGET_PRECISION:.0%}"
    )
else:
    # If no threshold meets the target, select the
    # highest-F1 threshold as a fallback.
    selected = results_df.sort_values(
        ["f1_score", "precision", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    selection_reason = (
        "No tested threshold met the precision target; "
        "selected highest F1-score as a fallback"
    )

selected_threshold = float(selected["threshold"])

# 5. Report results at the selected threshold
final_predictions = (
    probabilities >= selected_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, final_predictions, labels=[0, 1]
).ravel()

precision = precision_score(
    y_true, final_predictions, zero_division=0
)
recall = recall_score(
    y_true, final_predictions, zero_division=0
)
f1 = f1_score(
    y_true, final_predictions, zero_division=0
)
pr_auc = average_precision_score(y_true, probabilities)

print("=" * 65)
print("PRECISION AND FALSE-ALARM EVALUATION")
print("=" * 65)

print(f"Validation transactions: {len(y_true)}")
print(f"Actual high-risk labels: {int(y_true.sum())}")
print(f"Selected threshold: {selected_threshold:.2f}")
print(f"Selection reason: {selection_reason}")

print("\nSelected threshold metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print("\nConfusion matrix counts")
print(f"True positives: {tp}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True negatives: {tn}")
print(f"Transactions flagged: {tp + fp}")

print("\nThreshold comparison")
print(
    results_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "false_positives",
            "flagged_transactions",
        ]
    ].round(3).to_string(index=False)
)

# 6. Save the selected threshold for later application use
threshold_info = {
    "threshold": selected_threshold,
    "target_precision": TARGET_PRECISION,
    "achieved_precision": float(precision),
    "achieved_recall": float(recall),
    "achieved_f1": float(f1),
    "validation_pr_auc": float(pr_auc),
    "selection_reason": selection_reason,
    "synthetic_labels": True,
}

with open(
    f"{REPORT_DIR}/risk_threshold.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(threshold_info, f, indent=2)

# 7. Plot the precision-recall curve
curve_precision, curve_recall, curve_thresholds = (
    precision_recall_curve(y_true, probabilities)
)

plt.figure(figsize=(8, 5))
plt.plot(curve_recall, curve_precision)
plt.axhline(
    TARGET_PRECISION,
    linestyle="--",
    label="Target precision",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{REPORT_DIR}/precision_recall_curve.png",
    dpi=150,
)
plt.show()

print("\nSaved:")
print(f"{REPORT_DIR}/threshold_comparison.csv")
print(f"{REPORT_DIR}/risk_threshold.json")
print(f"{REPORT_DIR}/precision_recall_curve.png")
