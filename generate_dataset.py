
import os
import numpy as np
import pandas as pd

# Make the generated dataset reproducible
np.random.seed(42)

NUM_TRANSACTIONS = 5000
NUM_EMPLOYEES = 200

departments = [
    "Sales", "IT", "HR", "Finance", "Operations"
]

categories = [
    "Travel", "Meals", "Accommodation",
    "Office Supplies", "Equipment"
]

payment_methods = [
    "Corporate Card", "Reimbursement", "Bank Transfer"
]

# Typical expense ranges in INR for the synthetic dataset
category_ranges = {
    "Travel": (1500, 12000),
    "Meals": (200, 2500),
    "Accommodation": (2000, 10000),
    "Office Supplies": (100, 3000),
    "Equipment": (1500, 20000)
}

os.makedirs("data", exist_ok=True)

records = []

for i in range(1, NUM_TRANSACTIONS + 1):
    department = np.random.choice(departments)
    category = np.random.choice(categories)
    payment_method = np.random.choice(payment_methods)

    min_amount, max_amount = category_ranges[category]

    amount = round(
        np.random.uniform(min_amount, max_amount), 2
    )

    transaction_date = pd.Timestamp(
        "2025-01-01"
    ) + pd.Timedelta(
        days=int(np.random.randint(0, 365))
    )

    # Simulated unusual-expense indicators
    is_duplicate = int(np.random.random() < 0.025)
    is_weekend = int(transaction_date.dayofweek >= 5)

    # Some transactions have unusually high amounts
    unusually_high = np.random.random() < 0.04

    if unusually_high:
        amount = round(amount * np.random.uniform(3, 8), 2)

    # Simulated policy violation
    policy_violation = int(
        (amount > max_amount * 2)
        or (category == "Equipment" and amount > 50000)
    )

    # Synthetic label: high-risk if one or more risk signals exist
    high_risk = int(
        is_duplicate == 1
        or policy_violation == 1
        or amount > max_amount * 2
    )

    records.append({
        "transaction_id": f"TXN{i:05d}",
        "employee_id": (
            f"EMP{np.random.randint(1, NUM_EMPLOYEES + 1):03d}"
        ),
        "department": department,
        "expense_category": category,
        "amount": amount,
        "transaction_date": transaction_date.strftime("%Y-%m-%d"),
        "payment_method": payment_method,
        "is_weekend": is_weekend,
        "is_duplicate": is_duplicate,
        "policy_violation": policy_violation,
        "high_risk": high_risk
    })

df = pd.DataFrame(records)

output_path = "data/expense_transactions.csv"
df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"File saved to: {output_path}")
print(f"Dataset shape: {df.shape}")

print("\nFirst five transactions:")
print(df.head())

print("\nHigh-risk label distribution:")
print(df["high_risk"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
