
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/expense_transactions.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found: {file_path}. "
        "Run generate_dataset.py first."
    )

df = pd.read_csv(file_path)
df["transaction_date"] = pd.to_datetime(
    df["transaction_date"], errors="coerce"
)

sns.set_theme(style="whitegrid")
os.makedirs("reports", exist_ok=True)

print("=" * 60)
print("CORPORATE EXPENSE PATTERN ANALYSIS")
print("=" * 60)

# 1. Basic dataset information
print("\n1. DATASET INFORMATION")
print(f"Total transactions: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Total expense amount: Rs. {df['amount'].sum():,.2f}")
print(f"Average expense: Rs. {df['amount'].mean():,.2f}")
print(f"Median expense: Rs. {df['amount'].median():,.2f}")

# 2. Statistical summary
print("\n2. EXPENSE DISTRIBUTION")
print(df["amount"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
).round(2))

# 3. Category-level spending
print("\n3. SPENDING BY EXPENSE CATEGORY")
category_summary = (
    df.groupby("expense_category")["amount"]
    .agg(["count", "mean", "median", "std", "sum"])
    .round(2)
    .sort_values("median", ascending=False)
)
print(category_summary)

category_summary.to_csv(
    "reports/category_summary.csv"
)

# 4. Department-level spending
print("\n4. SPENDING BY DEPARTMENT")
department_summary = (
    df.groupby("department")["amount"]
    .agg(["count", "mean", "median", "std", "sum"])
    .round(2)
    .sort_values("median", ascending=False)
)
print(department_summary)

department_summary.to_csv(
    "reports/department_summary.csv"
)

# 5. Spending by payment method
print("\n5. SPENDING BY PAYMENT METHOD")
print(
    df.groupby("payment_method")["amount"]
    .agg(["count", "mean", "median", "sum"])
    .round(2)
)

# 6. High-risk label distribution
print("\n6. SYNTHETIC LABEL DISTRIBUTION")
print(df["high_risk"].value_counts())
print(
    "High-risk percentage: "
    f"{df['high_risk'].mean() * 100:.2f}%"
)

# 7. Missing data and duplicate IDs
print("\n7. DATA QUALITY")
print("Missing values:")
print(df.isnull().sum())
print(
    "Duplicate transaction IDs:",
    df["transaction_id"].duplicated().sum()
)

# 8. Histogram: overall expense distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["amount"], bins=50, kde=True)
plt.title("Distribution of Corporate Expense Amounts")
plt.xlabel("Expense Amount (INR)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("reports/expense_distribution.png", dpi=150)
plt.show()

# 9. Box plot: spending by category
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="expense_category",
    y="amount"
)
plt.title("Expense Distribution by Category")
plt.xlabel("Expense Category")
plt.ylabel("Expense Amount (INR)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/category_boxplot.png", dpi=150)
plt.show()

# 10. Bar chart: median expense by department
plt.figure(figsize=(9, 5))
sns.barplot(
    data=df,
    x="department",
    y="amount",
    estimator="median",
    errorbar=None
)
plt.title("Median Expense by Department")
plt.xlabel("Department")
plt.ylabel("Median Expense (INR)")
plt.tight_layout()
plt.savefig("reports/department_median.png", dpi=150)
plt.show()

# 11. Monthly transaction patterns
monthly = (
    df.dropna(subset=["transaction_date"])
    .set_index("transaction_date")
    .resample("ME")
    .agg(
        transaction_count=("amount", "size"),
        total_amount=("amount", "sum")
    )
)

print("\n8. MONTHLY SPENDING")
print(monthly.round(2))

monthly.to_csv("reports/monthly_summary.csv")

plt.figure(figsize=(10, 5))
monthly["total_amount"].plot(marker="o")
plt.title("Monthly Corporate Expense Totals")
plt.xlabel("Month")
plt.ylabel("Total Expense (INR)")
plt.tight_layout()
plt.savefig("reports/monthly_spending.png", dpi=150)
plt.show()

print("\nAnalysis completed successfully.")
print("Charts and summary CSV files are saved in reports/.")
