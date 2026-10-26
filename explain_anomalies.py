
import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/expense_risk_model.joblib"
VALIDATION_PATH = "data/splits/validation.csv"
THRESHOLD_PATH = "reports/risk_threshold.json"

REPORT_DIR = "reports/shap"
os.makedirs(REPORT_DIR, exist_ok=True)

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
]

# 1. Load the saved model and validation data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(VALIDATION_PATH)

y_true = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
X = X.drop(columns=EXCLUDED_COLUMNS, errors="ignore")

# 2. Load the selected threshold
with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_info = json.load(f)

threshold = float(threshold_info["threshold"])

# 3. Separate preprocessing from the classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# 4. Transform the validation data using the fitted preprocessor
X_transformed = preprocessor.transform(X)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X.index,
)

# 5. Predict risk probabilities
probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# 6. Select a small background and explanation sample
# The background represents the model's reference inputs.
background = X_transformed.sample(
    n=min(40, len(X_transformed)),
    random_state=42,
)

flagged_positions = np.flatnonzero(predictions == 1)

if len(flagged_positions) == 0:
    print("No transactions were flagged at the selected threshold.")
    print("Try a lower threshold for exploratory analysis.")
    raise SystemExit(0)

# Explain a small number of flagged cases to keep runtime manageable.
selected_positions = flagged_positions[
    np.argsort(probabilities[flagged_positions])[::-1]
][:10]

X_explain = X_transformed.iloc[selected_positions]

# 7. Create model-agnostic SHAP explanations.
# This works with either Logistic Regression or Random Forest.
explainer = shap.Explainer(
    classifier.predict_proba,
    background,
    feature_names=list(feature_names),
    seed=42,
)

shap_values = explainer(
    X_explain,
    max_evals=2 * X_transformed.shape[1] + 1,
)

# 8. Extract SHAP values for the high-risk probability (class 1)
values = shap_values.values

if values.ndim == 3:
    positive_values = values[:, :, 1]
else:
    positive_values = values

base_values = np.asarray(shap_values.base_values)

if base_values.ndim == 2:
    positive_base_values = base_values[:, 1]
else:
    positive_base_values = base_values.reshape(-1)

# 9. Global importance for the explained sample
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(positive_values).mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(
    f"{REPORT_DIR}/global_feature_importance.csv",
    index=False,
)

print("=" * 65)
print("EXPLAINABLE AI - SHAP ANALYSIS")
print("=" * 65)
print(f"Selected threshold: {threshold:.2f}")
print(f"Flagged validation transactions: {len(flagged_positions)}")

print("\nMost influential features in the explained sample:")
print(importance.head(10).to_string(index=False))

# 10. Global feature importance plot
plt.figure()
shap.summary_plot(
    positive_values,
    X_explain,
    plot_type="bar",
    max_display=15,
    show=False,
)
plt.tight_layout()
plt.savefig(
    f"{REPORT_DIR}/shap_global_importance.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 11. Detailed SHAP summary plot
plt.figure()
shap.summary_plot(
    positive_values,
    X_explain,
    max_display=15,
    show=False,
)
plt.tight_layout()
plt.savefig(
    f"{REPORT_DIR}/shap_summary.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 12. Explain the highest-scoring flagged transaction
best_position = int(selected_positions[0])
transaction_id = df.iloc[best_position].get(
    "transaction_id", f"row_{best_position}"
)

waterfall_explanation = shap.Explanation(
    values=positive_values[0],
    base_values=float(positive_base_values[0]),
    data=X_explain.iloc[0].to_numpy(),
    feature_names=list(feature_names),
)

plt.figure()
shap.plots.waterfall(
    waterfall_explanation,
    max_display=12,
    show=False,
)
plt.tight_layout()
plt.savefig(
    f"{REPORT_DIR}/shap_transaction_explanation.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 13. Save transaction-level prediction and explanation summary
local_rows = []

for i, position in enumerate(selected_positions):
    contributions = positive_values[i]

    top_indices = np.argsort(
        np.abs(contributions)
    )[::-1][:5]

    top_factors = []

    for j in top_indices:
        direction = (
            "increased"
            if contributions[j] > 0
            else "decreased"
        )

        top_factors.append(
            f"{feature_names[j]} {direction} risk "
            f"(SHAP={contributions[j]:.4f})"
        )

    local_rows.append({
        "transaction_id": df.iloc[position].get(
            "transaction_id", f"row_{position}"
        ),
        "predicted_risk_probability": float(
            probabilities[position]
        ),
        "selected_threshold": threshold,
        "predicted_high_risk": int(predictions[position]),
        "synthetic_actual_label": int(y_true.iloc[position]),
        "top_shap_factors": " | ".join(top_factors),
    })

pd.DataFrame(local_rows).to_csv(
    f"{REPORT_DIR}/transaction_explanations.csv",
    index=False,
)

print("\nSaved SHAP reports:")
print(f"{REPORT_DIR}/global_feature_importance.csv")
print(f"{REPORT_DIR}/shap_global_importance.png")
print(f"{REPORT_DIR}/shap_summary.png")
print(f"{REPORT_DIR}/shap_transaction_explanation.png")
print(f"{REPORT_DIR}/transaction_explanations.csv")
