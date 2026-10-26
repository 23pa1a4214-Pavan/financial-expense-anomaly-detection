
import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = "models/expense_risk_model.joblib"
THRESHOLD_PATH = "reports/risk_threshold.json"

st.set_page_config(
    page_title="Corporate Expense Risk Audit",
    page_icon="📊",
    layout="wide",
)

st.title("Financial Expense Anomaly Detection")
st.caption(
    "Machine learning-based expense risk screening "
    "and audit prioritization"
)

# --------------------------------------------------
# 1. Load the saved model and threshold
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_info = json.load(f)

    return model, float(threshold_info["threshold"])


if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

if not os.path.exists(THRESHOLD_PATH):
    st.error("Risk threshold not found. Run evaluate_precision.py first.")
    st.stop()

model, threshold = load_model()

st.info(
    f"Current classification threshold: {threshold:.2f}. "
    "Predictions are screening indicators, not proof of fraud."
)

# --------------------------------------------------
# 2. Input validation and model features
# --------------------------------------------------

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
    "audit_label",
]

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


def prepare_features(df):
    df = df.copy()

    # Convert dates and generate time-based features
    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    )

    if df["transaction_date"].isna().any():
        raise ValueError("Invalid transaction dates found.")

    df["transaction_month"] = df["transaction_date"].dt.month
    df["transaction_day"] = df["transaction_date"].dt.day
    df["transaction_day_of_week"] = (
        df["transaction_date"].dt.dayofweek
    )
    df["transaction_day_of_year"] = (
        df["transaction_date"].dt.dayofyear
    )

    # Employee transaction count within the uploaded batch
    df["employee_transaction_count"] = (
        df.groupby("employee_id")["transaction_id"]
        .transform("count")
    )

    # Match the percentile feature used during training.
    # For a single uploaded transaction, this is not a reliable
    # historical percentile; see the limitation below.
    df["above_category_p95"] = 0

    expected = list(
        model.named_steps["preprocessor"].feature_names_in_
    )

    missing = set(expected) - set(df.columns)

    if missing:
        raise ValueError(
            "Could not generate required model features: "
            + ", ".join(sorted(missing))
        )

    return df[expected]



def score_transactions(uploaded_df):
    df = uploaded_df.copy()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["transaction_id"] = (
        df["transaction_id"].astype("string").str.strip()
    )

    if df["transaction_id"].isna().any() or (
        df["transaction_id"] == ""
    ).any():
        raise ValueError("Transaction IDs cannot be empty.")

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "Duplicate transaction IDs detected. "
            "Please resolve them before scoring."
        )

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    if df["amount"].isna().any() or (
        ~np.isfinite(df["amount"])
    ).any() or (df["amount"] <= 0).any():
        raise ValueError(
            "Amounts must be valid positive numeric values."
        )

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], errors="coerce"
    )

    if df["transaction_date"].isna().any():
        raise ValueError(
            "One or more transaction dates are invalid."
        )

    # Recompute the date-derived feature.
    df["is_weekend"] = (
        df["transaction_date"].dt.dayofweek >= 5
    ).astype(int)

    # Ensure category values match the training schema.
    for col in [
        "department",
        "expense_category",
        "payment_method",
    ]:
        df[col] = df[col].astype("string").str.strip()

        if df[col].isna().any() or (
            df[col] == ""
        ).any():
            raise ValueError(
                f"Missing values found in {col}."
            )

    X = prepare_features(df)

    # Model probabilities and classification
    df["risk_probability"] = model.predict_proba(X)[:, 1]

    df["model_high_risk"] = (
        df["risk_probability"] >= threshold
    ).astype(int)

    # Potential claim exposure, not confirmed loss
    df["potential_exposure_inr"] = np.where(
        df["model_high_risk"] == 1,
        df["amount"],
        0.0,
    )

    # Transparent prioritization heuristic
    df["amount_percentile"] = (
        df["amount"].rank(pct=True) * 100
    )

    duplicate = (
        pd.to_numeric(df["is_duplicate"], errors="coerce")
        if "is_duplicate" in df.columns
        else pd.Series(0, index=df.index)
    )

    violation = (
        pd.to_numeric(df["policy_violation"], errors="coerce")
        if "policy_violation" in df.columns
        else pd.Series(0, index=df.index)
    )

    statistical = (
        pd.to_numeric(df["statistical_anomaly"], errors="coerce")
        if "statistical_anomaly" in df.columns
        else pd.Series(0, index=df.index)
    )

    duplicate = duplicate.fillna(0).clip(0, 1)
    violation = violation.fillna(0).clip(0, 1)
    statistical = statistical.fillna(0).clip(0, 1)

    df["additional_warning_score"] = (
        0.4 * duplicate + 0.4 * violation + 0.2 * statistical
    )

    df["priority_score"] = (
        70 * df["risk_probability"]
        + 0.20 * df["amount_percentile"]
        + 10 * df["additional_warning_score"]
    )

    def assign_priority(row):
        if row["model_high_risk"] == 1:
            if (
                row["risk_probability"] >= 0.85
                and (
                    row["amount_percentile"] >= 95
                    or row["additional_warning_score"] >= 0.4
                )
            ):
                return "P1 - Urgent"

            return "P2 - High"

        if (
            row["amount_percentile"] >= 95
            or row["additional_warning_score"] >= 0.4
        ):
            return "P3 - Routine"

        return "Monitor"

    df["audit_priority"] = df.apply(assign_priority, axis=1)

    priority_order = {
        "P1 - Urgent": 0,
        "P2 - High": 1,
        "P3 - Routine": 2,
        "Monitor": 3,
    }

    df["priority_order"] = df["audit_priority"].map(
        priority_order
    )

    df = df.sort_values(
        ["priority_order", "priority_score", "amount"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# 3. Sidebar
# --------------------------------------------------

st.sidebar.header("Audit Controls")
st.sidebar.write(f"Model threshold: {threshold:.2f}")
st.sidebar.caption(
    "Use the same input schema as the training dataset."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload expense transaction CSV",
    type=["csv"],
)

# --------------------------------------------------
# 4. Upload and score
# --------------------------------------------------

if uploaded_file is None:
    st.info(
        "Upload a CSV file containing transaction records "
        "to begin the analysis."
    )

    st.subheader("Required input columns")
    st.code(
        "transaction_id, employee_id, department, "
        "expense_category, amount, transaction_date, "
        "payment_method, is_weekend"
    )

    st.caption(
        "Use the cleaned or engineered project dataset as a "
        "starting point. Extra columns are permitted."
    )
    st.stop()

try:
    input_df = pd.read_csv(uploaded_file)
    result = score_transactions(input_df)
except Exception as exc:
    st.error(f"Could not analyze the uploaded file: {exc}")
    st.stop()

# --------------------------------------------------
# 5. Dashboard metrics
# --------------------------------------------------

total = len(result)
flagged = int(result["model_high_risk"].sum())
flagged_amount = float(
    result.loc[
        result["model_high_risk"] == 1, "amount"
    ].sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions Analyzed", f"{total:,}")
c2.metric("High-Risk Flags", f"{flagged:,}")
c3.metric(
    "Flag Rate",
    f"{100 * flagged / total:.1f}%" if total else "0%",
)
c4.metric(
    "Flagged Claim Amount",
    f"₹{flagged_amount:,.0f}",
)

st.caption(
    "Flagged claim amount is the value of transactions selected "
    "for review, not confirmed financial loss."
)

# --------------------------------------------------
# 6. Charts
# --------------------------------------------------

st.subheader("Expense Risk Overview")

left, right = st.columns(2)

with left:
    counts = (
        result["audit_priority"]
        .value_counts()
        .reindex(
            [
                "P1 - Urgent",
                "P2 - High",
                "P3 - Routine",
                "Monitor",
            ],
            fill_value=0,
        )
    )

    st.bar_chart(counts)

with right:
    category_amounts = (
        result.groupby("expense_category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_amounts)

# --------------------------------------------------
# 7. Transaction results
# --------------------------------------------------

st.subheader("Audit Priority Register")

priority_filter = st.multiselect(
    "Filter by audit priority",
    options=[
        "P1 - Urgent",
        "P2 - High",
        "P3 - Routine",
        "Monitor",
    ],
    default=["P1 - Urgent", "P2 - High"],
)

filtered = result[
    result["audit_priority"].isin(priority_filter)
]

display_columns = [
    "transaction_id",
    "department",
    "expense_category",
    "amount",
    "risk_probability",
    "audit_priority",
    "priority_score",
]

st.dataframe(
    filtered[display_columns].style.format({
        "amount": "₹{:,.2f}",
        "risk_probability": "{:.1%}",
        "priority_score": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Audit Priority CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="audit_priority_register.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 8. SHAP explanation for a selected transaction
# --------------------------------------------------

st.subheader("Explain a Transaction with SHAP")

if len(result) > 0:
    transaction_options = result["transaction_id"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select a transaction",
        transaction_options,
    )

    if st.button("Generate SHAP Explanation"):
        try:
            row_index = result.index[
                result["transaction_id"].astype(str) == selected_id
            ][0]

            raw_row = result.loc[[row_index]]
            X_row = prepare_features(raw_row)

            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            transformed_row = preprocessor.transform(X_row)

            if hasattr(transformed_row, "toarray"):
                transformed_row = transformed_row.toarray()

            feature_names = (
                preprocessor.get_feature_names_out()
            )

            X_transformed = pd.DataFrame(
                transformed_row,
                columns=feature_names,
            )

            # Use a small, deterministic background sample
            background_raw = input_df.copy()

            # Ensure the background has the same required features
            background_raw["transaction_date"] = pd.to_datetime(
                background_raw["transaction_date"],
                errors="coerce",
            )
            background_raw["is_weekend"] = (
                background_raw["transaction_date"]
                .dt.dayofweek.ge(5)
                .astype(int)
            )

            background_X = prepare_features(background_raw)
            background_transformed = preprocessor.transform(
                background_X
            )

            if hasattr(background_transformed, "toarray"):
                background_transformed = (
                    background_transformed.toarray()
                )

            background_df = pd.DataFrame(
                background_transformed,
                columns=feature_names,
            ).sample(
                n=min(30, len(background_transformed)),
                random_state=42,
            )

            explainer = shap.Explainer(
                classifier.predict_proba,
                background_df,
                feature_names=list(feature_names),
                seed=42,
            )

            explanation = explainer(
                X_transformed,
                max_evals=2 * len(feature_names) + 1,
            )

            values = explanation.values

            if values.ndim == 3:
                positive_values = values[0, :, 1]
                base_values = explanation.base_values[0, 1]
            else:
                positive_values = values[0]
                base_values = np.asarray(
                    explanation.base_values
                ).reshape(-1)[0]

            local_importance = pd.DataFrame({
                "Feature": feature_names,
                "SHAP contribution": positive_values,
            })

            local_importance["Absolute contribution"] = (
                local_importance["SHAP contribution"].abs()
            )

            local_importance = local_importance.sort_values(
                "Absolute contribution",
                ascending=False,
            ).head(10)

            st.write(
                "Positive SHAP values increase the model's "
                "high-risk probability; negative values decrease it."
            )

            st.dataframe(
                local_importance.drop(
                    columns=["Absolute contribution"]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "SHAP explanations describe model behavior, "
                "not proof of fraud or causation."
            )

        except Exception as exc:
            st.error(f"SHAP explanation failed: {exc}")

# --------------------------------------------------
# 9. Audit disclaimer
# --------------------------------------------------

st.divider()

st.warning(
    "This prototype uses synthetic training labels. "
    "Predictions are for demonstration and audit screening only. "
    "Verify expenses against receipts, approvals, and company policy "
    "before taking action."
)
