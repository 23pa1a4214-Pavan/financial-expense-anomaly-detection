
import os
import pandas as pd

INPUT_FILE = "data/expense_transactions.csv"
OUTPUT_FILE = "data/cleaned_expense_transactions.csv"

REQUIRED_COLUMNS = [
    "transaction_id",
    "employee_id",
    "department",
    "expense_category",
    "amount",
    "transaction_date",
    "payment_method",
    "is_weekend",
    "is_duplicate",
    "policy_violation",
    "high_risk",
]

os.makedirs("data", exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(INPUT_FILE)
initial_rows = len(df)

print("=" * 60)
print("EXPENSE DATA CLEANING")
print("=" * 60)
print(f"Initial rows: {initial_rows}")

# 2. Check required columns
missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# 3. Remove surrounding spaces from text fields
text_columns = [
    "transaction_id",
    "employee_id",
    "department",
    "expense_category",
    "payment_method",
]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# Treat empty text fields as missing
df[text_columns] = df[text_columns].replace("", pd.NA)

# 4. Convert numeric fields
numeric_columns = [
    "amount",
    "is_weekend",
    "is_duplicate",
    "policy_violation",
    "high_risk",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Convert transaction dates
df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

# 6. Report missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 7. Remove rows missing essential information
essential_columns = [
    "transaction_id",
    "employee_id",
    "department",
    "expense_category",
    "amount",
    "transaction_date",
    "payment_method",
    "is_weekend",
    "is_duplicate",
    "policy_violation",
    "high_risk",
]

df = df.dropna(subset=essential_columns).copy()

# 8. Remove duplicate transaction IDs
df = df.drop_duplicates(
    subset=["transaction_id"], keep="first"
).copy()

# 9. Remove invalid transaction amounts
df = df[df["amount"] > 0].copy()

# 10. Validate binary indicator columns
binary_columns = [
    "is_weekend",
    "is_duplicate",
    "policy_violation",
    "high_risk",
]

for col in binary_columns:
    df = df[df[col].isin([0, 1])].copy()
    df[col] = df[col].astype(int)

# 11. Validate category values
valid_departments = {
    "Sales", "IT", "HR", "Finance", "Operations"
}

valid_categories = {
    "Travel", "Meals", "Accommodation",
    "Office Supplies", "Equipment"
}

valid_payment_methods = {
    "Corporate Card", "Reimbursement", "Bank Transfer"
}

df = df[df["department"].isin(valid_departments)].copy()
df = df[df["expense_category"].isin(valid_categories)].copy()
df = df[df["payment_method"].isin(valid_payment_methods)].copy()

# 12. Recalculate weekend flag from the transaction date
df["is_weekend"] = (
    df["transaction_date"].dt.dayofweek >= 5
).astype(int)

# 13. Restore consistent date format
df["transaction_date"] = (
    df["transaction_date"].dt.strftime("%Y-%m-%d")
)

# 14. Save cleaned data
df.to_csv(OUTPUT_FILE, index=False)

# 15. Final validation report
print("\nCLEANING SUMMARY")
print(f"Initial rows: {initial_rows}")
print(f"Final rows: {len(df)}")
print(f"Rows removed: {initial_rows - len(df)}")
print(f"Final columns: {len(df.columns)}")
print(f"Invalid amounts: {(df['amount'] <= 0).sum()}")
print(
    "Duplicate transaction IDs:",
    df["transaction_id"].duplicated().sum()
)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")
