
import pandas as pd

REQUIRED_COLUMNS = [
    "transaction_id",
    "employee_id",
    "department",
    "expense_category",
    "amount",
    "transaction_date",
    "payment_method",
    "is_weekend",
]


def validate_transactions(df):
    if df.empty:
        raise ValueError("The uploaded CSV contains no transactions.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["transaction_id"].isna().any():
        raise ValueError("Transaction IDs cannot be empty.")

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs detected.")

    amount = pd.to_numeric(df["amount"], errors="coerce")

    if amount.isna().any():
        raise ValueError("Amounts must be numeric.")

    if (amount <= 0).any():
        raise ValueError("Amounts must be positive.")

    dates = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    )

    if dates.isna().any():
        raise ValueError("Invalid transaction dates.")

    return True


# Test with the cleaned dataset
df = pd.read_csv("data/cleaned_expense_transactions.csv")

assert validate_transactions(df)
print("PASS: Valid dataset")

# Test missing amount column
try:
    validate_transactions(df.drop(columns=["amount"]))
    raise AssertionError("Missing amount was not rejected.")
except ValueError as e:
    print("PASS: Missing column rejected:", e)

# Test invalid amount
bad_amount = df.head(2).copy()
bad_amount["amount"] = bad_amount["amount"].astype(object)
bad_amount.loc[bad_amount.index[0], "amount"] = "abc"

try:
    validate_transactions(bad_amount)
    raise AssertionError("Invalid amount was not rejected.")
except ValueError as e:
    print("PASS: Invalid amount rejected:", e)

# Test negative amount
bad_amount = df.head(2).copy()
bad_amount.loc[bad_amount.index[0], "amount"] = -500

try:
    validate_transactions(bad_amount)
    raise AssertionError("Negative amount was not rejected.")
except ValueError as e:
    print("PASS: Negative amount rejected:", e)

# Test duplicate transaction ID
bad_ids = pd.concat([df.head(1), df.head(1)], ignore_index=True)

try:
    validate_transactions(bad_ids)
    raise AssertionError("Duplicate IDs were not rejected.")
except ValueError as e:
    print("PASS: Duplicate transaction IDs rejected:", e)

# Test invalid date
bad_date = df.head(2).copy()
bad_date.loc[bad_date.index[0], "transaction_date"] = "invalid-date"

try:
    validate_transactions(bad_date)
    raise AssertionError("Invalid date was not rejected.")
except ValueError as e:
    print("PASS: Invalid date rejected:", e)

print("\nInput validation tests completed.")
