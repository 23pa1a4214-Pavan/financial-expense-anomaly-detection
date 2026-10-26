
import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data/splits"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET = "audit_label"

# --------------------------------------------------
# 1. Load the data splits
# --------------------------------------------------

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/validation.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET].astype(int)

if y_train.nunique() < 2:
    raise ValueError("Training data must contain both classes.")

if y_val.nunique() < 2:
    raise ValueError("Validation data must contain both classes.")

# Remove any remaining identifier or label-related fields.
excluded = [
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

X_train = X_train.drop(columns=excluded, errors="ignore")
X_val = X_val.drop(columns=excluded, errors="ignore")

# --------------------------------------------------
# 2. Identify numerical and categorical columns
# --------------------------------------------------

categorical_cols = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

print("=" * 65)
print("SUPERVISED EXPENSE RISK CLASSIFICATION")
print("=" * 65)

print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Numerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")

# --------------------------------------------------
# 3. Create preprocessing pipelines
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
fitted_models = {}

# Balanced sample weights help address class imbalance.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

# --------------------------------------------------
# 5. Train and evaluate on validation data
# --------------------------------------------------

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    # Each model gets its own fitted preprocessing pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    if name == "Random Forest":
        pipeline.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights,
        )
    else:
        pipeline.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weights,
        )

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_val, probabilities
        ),
    }

    results.append(metrics)
    fitted_models[name] = pipeline

    print("\nValidation metrics:")
    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Low risk", "High risk"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

# --------------------------------------------------
# 6. Select a model using validation PR-AUC
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "pr_auc", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_model = fitted_models[best_name]

# Save the fitted preprocessing + classifier pipeline.
model_path = f"{MODEL_DIR}/expense_risk_model.joblib"
joblib.dump(best_model, model_path)

results_df.to_csv(
    f"{REPORT_DIR}/model_comparison.csv",
    index=False,
)

with open(
    f"{REPORT_DIR}/best_model_info.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "selected_model": best_name,
            "selection_metric": "validation PR-AUC",
            "target": TARGET,
            "synthetic_labels": True,
            "excluded_leakage_features": excluded,
            "input_features": X_train.columns.tolist(),
        },
        f,
        indent=2,
    )

print(f"\nSelected model: {best_name}")
print(f"Saved model: {model_path}")
print(f"Comparison report: {REPORT_DIR}/model_comparison.csv")
print("\nTraining and validation completed.")
print("The test dataset has not been used.")
