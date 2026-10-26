
import os
import numpy as np
import pandas as pd

INPUT_FILE = "data/cleaned_expense_transactions.csv"
OUTPUT_FILE = "data/expense_baseline_results.csv"
SUMMARY_FILE = "reports/statistical_baseline_summary.csv"

os.makedirs("reports", exist_ok=True)

# 1. Load cleaned transactions
df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("STATISTICAL BASELINE ANALYSIS")
print("=" * 65)
print(f"Transactions loaded: {len(df)}")

# 2. Calculate baseline statistics by expense category
baseline = (
    df.groupby("expense_category")["amount"]
    .agg(
        transaction_count="count",
        mean_amount="mean",
        median_amount="median",
        standard_deviation="std",
        percentile_95=lambda x: x.quantile(0.95),
        percentile_99=lambda x: x.quantile(0.99),
    )
    .reset_index()
)

# 3. Calculate Median Absolute Deviation (MAD)
category_median = df.groupby(
    "expense_category"
)["amount"].transform("median")

absolute_deviation = (df["amount"] - category_median).abs()

mad = absolute_deviation.groupby(
    df["expense_category"]
).transform("median")

# 4. Add category baseline features to each transaction
df["category_median"] = category_median
df["category_mad"] = mad

df["amount_to_median_ratio"] = (
    df["amount"] / df["category_median"]
)

# 5. Calculate robust z-score
# If MAD is zero, the score is undefined; mark it as zero
# and rely on percentile/amount checks as additional signals.
df["robust_z_score"] = np.where(
    df["category_mad"] > 0,
    0.6745
    * (df["amount"] - df["category_median"])
    / df["category_mad"],
    0.0,
)

# 6. Create statistical screening flags
df["above_category_p95"] = (
    df["amount"]
    > df["expense_category"].map(
        df.groupby("expense_category")["amount"].quantile(0.95)
    )
).astype(int)

df["statistical_anomaly"] = (
    (df["robust_z_score"].abs() > 3.5)
    | (
        df["amount"]
        > df["expense_category"].map(
            df.groupby("expense_category")["amount"].quantile(0.99)
        )
    )
).astype(int)

# 7. Save category summary
baseline = baseline.round(2)
baseline.to_csv(SUMMARY_FILE, index=False)

# 8. Save transaction-level results
df.to_csv(OUTPUT_FILE, index=False)

# 9. Display results
print("\nCATEGORY BASELINES")
print(baseline.to_string(index=False))

print("\nSTATISTICAL SCREENING RESULTS")
print(
    df["statistical_anomaly"]
    .value_counts()
    .rename(index={0: "Not flagged", 1: "Flagged"})
    .to_string()
)

print(
    "\nTransactions flagged:",
    int(df["statistical_anomaly"].sum())
)

print("\nEXAMPLE FLAGGED TRANSACTIONS")
columns = [
    "transaction_id",
    "expense_category",
    "amount",
    "category_median",
    "amount_to_median_ratio",
    "robust_z_score",
]

flagged = df[df["statistical_anomaly"] == 1]

print(
    flagged[columns]
    .sort_values("robust_z_score", ascending=False)
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\nSaved files:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)
