
import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/labeled_expense_transactions.csv"
OUTPUT_FILE = "data/engineered_expense_features.csv"

os.makedirs("data", exist_ok=True)

# 1. Load the labeled dataset
df = pd.read_csv(INPUT_FILE)

df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

print("=" * 65)
print("EXPENSE FEATURE ENGINEERING")
print("=" * 65)
print(f"Transactions loaded: {len(df)}")

# 2. Validate required fields
required_columns = [
    "transaction_id",
    "employee_id",
    "expense_category",
    "amount",
    "transaction_date",
    "audit_label",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if df["transaction_date"].isna().any():
    raise ValueError("Invalid transaction dates found.")

if (df["amount"] <= 0).any():
    raise ValueError("Non-positive transaction amounts found.")

# 3. Time-related features
df["transaction_month"] = df["transaction_date"].dt.month
df["transaction_day_of_week"] = (
    df["transaction_date"].dt.dayofweek
)
df["transaction_day"] = df["transaction_date"].dt.day
df["transaction_day_of_year"] = (
    df["transaction_date"].dt.dayofyear
)

# 4. Category-level features
# These are exploratory features for the current dataset.
# In model training, fit these statistics on training data only.
category_groups = df.groupby("expense_category")["amount"]

category_median = category_groups.transform("median")
category_mad = (
    df["amount"] - category_median
).abs().groupby(df["expense_category"]).transform("median")

df["category_median"] = category_median
df["category_percentile"] = (
    df.groupby("expense_category")["amount"]
    .rank(pct=True)
)

df["amount_to_median_ratio"] = (
    df["amount"] / category_median
)

df["robust_z_score"] = np.where(
    category_mad > 0,
    0.6745 * (df["amount"] - category_median) / category_mad,
    0.0,
)

# 5. Employee-level exploratory features
employee_groups = df.groupby("employee_id")["amount"]

df["employee_transaction_count"] = (
    df.groupby("employee_id")["transaction_id"]
    .transform("count")
)

df["employee_avg_amount"] = employee_groups.transform("mean")

df["amount_to_employee_avg_ratio"] = (
    df["amount"] / df["employee_avg_amount"]
)

# 6. Ensure binary indicators are integers
for col in ["is_weekend", "is_duplicate", "policy_violation"]:
    df[col] = df[col].astype(int)

# 7. Save the engineered dataset
df.to_csv(OUTPUT_FILE, index=False)

# 8. Report the new features
new_features = [
    "transaction_month",
    "transaction_day_of_week",
    "transaction_day",
    "transaction_day_of_year",
    "category_median",
    "category_percentile",
    "amount_to_median_ratio",
    "robust_z_score",
    "employee_transaction_count",
    "employee_avg_amount",
    "amount_to_employee_avg_ratio",
]

print("\nNew features created:")
for feature in new_features:
    print(f"- {feature}")

print(f"\nDataset shape: {df.shape}")
print(f"Output saved to: {OUTPUT_FILE}")

print("\nFeature preview:")
print(
    df[
        [
            "transaction_id",
            "amount",
            "category_median",
            "amount_to_median_ratio",
            "robust_z_score",
            "category_percentile",
            "audit_label",
        ]
    ].head(10).round(3).to_string(index=False)
)

print("\nMissing values in new features:")
print(df[new_features].isnull().sum())
