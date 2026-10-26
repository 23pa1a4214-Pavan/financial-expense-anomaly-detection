
import os
import pandas as pd

INPUT_FILE = "data/expense_baseline_results.csv"
OUTPUT_FILE = "data/labeled_expense_transactions.csv"

os.makedirs("data", exist_ok=True)

# 1. Load transaction data with statistical baseline results
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("HIGH-RISK TRANSACTION LABELING")
print("=" * 60)

print(f"Total transactions: {len(df)}")

# 2. Define labeling rules
# These labels are synthetic, not verified audit findings.
df["audit_label"] = (
    (df["is_duplicate"] == 1)
    | (df["policy_violation"] == 1)
    | (df["amount_to_median_ratio"] >= 3)
).astype(int)

# 3. Add an explanation for the synthetic label
def label_reason(row):
    reasons = []

    if row["is_duplicate"] == 1:
        reasons.append("Potential duplicate claim")

    if row["policy_violation"] == 1:
        reasons.append("Simulated policy violation")

    if row["amount_to_median_ratio"] >= 3:
        reasons.append("Amount at least 3x category median")

    if not reasons:
        return "No predefined synthetic risk signal"

    return "; ".join(reasons)


df["synthetic_label_reason"] = df.apply(
    label_reason, axis=1
)

# 4. Display label distribution
label_counts = (
    df["audit_label"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

print("\nAudit label distribution:")
print(f"Low risk: {label_counts[0]}")
print(f"High risk: {label_counts[1]}")
print(
    "High-risk percentage: "
    f"{df['audit_label'].mean() * 100:.2f}%"
)

# 5. Compare labels with statistical anomaly flags
print("\nComparison with statistical anomaly flags:")
print(
    pd.crosstab(
        df["audit_label"],
        df["statistical_anomaly"],
        rownames=["Synthetic audit label"],
        colnames=["Statistical anomaly flag"]
    )
)

# 6. Show examples of high-risk transactions
print("\nSample high-risk transactions:")
sample_columns = [
    "transaction_id",
    "expense_category",
    "amount",
    "amount_to_median_ratio",
    "robust_z_score",
    "audit_label",
    "synthetic_label_reason",
]

print(
    df.loc[df["audit_label"] == 1, sample_columns]
    .head(10)
    .round(2)
    .to_string(index=False)
)

# 7. Save labeled data
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nLabeled dataset saved to: {OUTPUT_FILE}")
print(f"Rows saved: {len(df)}")
