
import os
import pandas as pd

from sklearn.model_selection import train_test_split

INPUT_FILE = "data/engineered_expense_features.csv"
OUTPUT_DIR = "data/splits"

RANDOM_STATE = 42
TARGET = "audit_label"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the engineered dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# 2. Validate the target
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

if df[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

if df[TARGET].nunique() != 2:
    raise ValueError("Expected a binary target containing 0 and 1.")

# 3. Remove columns that must not be model inputs
# Some of these columns directly encode the synthetic labeling rules.
excluded_columns = [
    "transaction_id",
    "employee_id",
    "transaction_date",
    "audit_label",
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

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

y = df[TARGET].astype(int)

# 4. Split into 70% training and 30% temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y,
)

# 5. Divide the temporary data equally:
# 15% validation and 15% test overall
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

# 6. Save splits, retaining target labels
train_df = X_train.copy()
train_df[TARGET] = y_train

val_df = X_val.copy()
val_df[TARGET] = y_val

test_df = X_test.copy()
test_df[TARGET] = y_test

train_df.to_csv(
    f"{OUTPUT_DIR}/train.csv", index=False
)
val_df.to_csv(
    f"{OUTPUT_DIR}/validation.csv", index=False
)
test_df.to_csv(
    f"{OUTPUT_DIR}/test.csv", index=False
)

# 7. Display split information
print("\nSPLIT SIZES")
for name, part in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df),
]:
    print(
        f"{name}: {len(part)} rows "
        f"({len(part) / len(df) * 100:.1f}%)"
    )

print("\nHIGH-RISK LABEL DISTRIBUTION")
for name, part in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df),
]:
    print(f"\n{name}:")
    print(part[TARGET].value_counts().sort_index())
    print(
        "High-risk percentage:",
        f"{part[TARGET].mean() * 100:.2f}%"
    )

print("\nMODEL INPUT FEATURES")
print(X.columns.tolist())

print("\nFiles saved:")
print(f"{OUTPUT_DIR}/train.csv")
print(f"{OUTPUT_DIR}/validation.csv")
print(f"{OUTPUT_DIR}/test.csv")

print("\nData splitting completed successfully.")
