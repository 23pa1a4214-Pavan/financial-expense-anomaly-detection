
import os
import json
import hashlib
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

MODEL_PATH = "models/expense_risk_model.joblib"
TEST_PATH = "data/splits/test.csv"
THRESHOLD_PATH = "reports/risk_threshold.json"
INFO_PATH = "reports/best_model_info.json"

OUTPUT_PATH = "reports/final_test_evaluation.json"

TARGET = "audit_label"

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

# 1. Check required files
for path in [MODEL_PATH, TEST_PATH, THRESHOLD_PATH, INFO_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# 2. Verify model file integrity using a checksum
sha256 = hashlib.sha256()

with open(MODEL_PATH, "rb") as model_file:
    for chunk in iter(lambda: model_file.read(8192), b""):
        sha256.update(chunk)

model_checksum = sha256.hexdigest()

# 3. Load the saved model and metadata
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_info = json.load(f)

with open(INFO_PATH, "r", encoding="utf-8") as f:
    model_info = json.load(f)

threshold = float(threshold_info["threshold"])

if not 0 <= threshold <= 1:
    raise ValueError("Saved threshold must be between 0 and 1.")

if not hasattr(model, "predict_proba"):
    raise TypeError("Saved model does not support predict_proba.")

if not hasattr(model, "named_steps"):
    raise TypeError("Expected a fitted sklearn Pipeline.")

if "preprocessor" not in model.named_steps:
    raise ValueError("Pipeline is missing its preprocessor.")

if "classifier" not in model.named_steps:
    raise ValueError("Pipeline is missing its classifier.")

# 4. Load the untouched test set
test_df = pd.read_csv(TEST_PATH)

if TARGET not in test_df.columns:
    raise ValueError(f"Missing test target: {TARGET}")

y_test = test_df[TARGET].astype(int)

if y_test.nunique() < 2:
    raise ValueError("Test data must contain both target classes.")

X_test = test_df.drop(columns=[TARGET])
X_test = X_test.drop(
    columns=EXCLUDED_COLUMNS,
    errors="ignore"
)

# Ensure input columns match the saved model
expected_features = list(
    model.named_steps["preprocessor"].feature_names_in_
)

missing_features = set(expected_features) - set(X_test.columns)

if missing_features:
    raise ValueError(
        f"Missing model input features: {sorted(missing_features)}"
    )

X_test = X_test[expected_features]

# 5. Run the saved model
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# 6. Calculate final test metrics
metrics = {
    "test_transactions": int(len(y_test)),
    "selected_model": model_info["selected_model"],
    "threshold": threshold,
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "pr_auc": float(
        average_precision_score(y_test, probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "model_sha256": model_checksum,
    "synthetic_labels": True,
    "interpretation": (
        "Evaluation against synthetic labels only; "
        "not verified real-world fraud detection performance."
    ),
}

# 7. Save evaluation and verification details
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 8. Display results
print("=" * 65)
print("SAVED MODEL VERIFICATION")
print("=" * 65)

print("Model loaded successfully.")
print("Preprocessing pipeline verified.")
print("Classifier verified.")
print(f"Selected model: {model_info['selected_model']}")
print(f"Input features: {len(expected_features)}")
print(f"Saved threshold: {threshold:.2f}")
print(f"Model SHA-256: {model_checksum}")

print("\nFINAL TEST METRICS")
for name in [
    "accuracy", "precision", "recall", "f1_score", "pr_auc"
]:
    print(f"{name}: {metrics[name]:.4f}")

print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
print(np.array(metrics["confusion_matrix"]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Low risk", "High risk"],
        zero_division=0,
    )
)

print(f"\nEvaluation saved to: {OUTPUT_PATH}")
print("Model verification completed successfully.")
