import os
import json
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Loan Approval Classification System",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    required_files = [
        "models/preprocessor.joblib",
        "models/rf_baseline.joblib",
        "models/loan_ann_model.h5",
        "models/feature_schema.json",
        "reports/evaluation_metrics.json"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"Missing model artifacts: {missing}. Run `python src/train.py` first.")
        st.stop()

    preprocessor = joblib.load("models/preprocessor.joblib")
    rf_model = joblib.load("models/rf_baseline.joblib")
    ann_model = tf.keras.models.load_model("models/loan_ann_model.h5")

    with open("models/feature_schema.json", "r") as f:
        schema = json.load(f)
    with open("reports/evaluation_metrics.json", "r") as f:
        metrics = json.load(f)

    return preprocessor, rf_model, ann_model, schema, metrics


preprocessor, rf_model, ann_model, schema, metrics = load_artifacts()

st.title("Neural Network-Based Loan Approval Classification System")
st.caption("End-to-End Credit Underwriting | Artificial Neural Network (ANN) vs. Random Forest Baseline")

tab1, tab2 = st.tabs(["Live Loan Underwriting Simulator", "Model Benchmarks & Diagnostics"])

with tab1:
    st.subheader("1. Applicant Financial & Credit Profile")
    input_data = {}
    cols = st.columns(3)
    idx = 0

    for col_name in schema.get("cat_cols", []):
        with cols[idx % 3]:
            options = schema["cat_values"][col_name]
            input_data[col_name] = st.selectbox(col_name, options)
        idx += 1

    for col_name in schema.get("raw_num_cols", []):
        with cols[idx % 3]:
            default_val = float(schema["num_medians"][col_name])
            input_data[col_name] = st.number_input(col_name, value=default_val)
        idx += 1

    st.divider()
    st.subheader("2. Approval Policy Threshold")
    threshold = st.slider(
        "Minimum Approval Probability Cutoff (Raise to tighten credit risk, lower to expand approvals)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    if st.button("Evaluate Loan Application", type="primary"):
        df_in = pd.DataFrame([input_data])

        # Replicate domain feature engineering from src/train.py
        if "loan_amnt" in df_in.columns and "person_income" in df_in.columns:
            df_in["calc_debt_to_income"] = df_in["loan_amnt"] / (df_in["person_income"] + 1.0)
        if "loan_amnt" in df_in.columns and "loan_int_rate" in df_in.columns:
            df_in["estimated_annual_interest"] = df_in["loan_amnt"] * (df_in["loan_int_rate"] / 100.0)

        X_proc = preprocessor.transform(df_in)
        ann_prob = float(ann_model.predict(X_proc, verbose=0)[0][0])
        rf_prob = float(rf_model.predict_proba(X_proc)[0][1])

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Primary Model: Artificial Neural Network")
            st.metric("Approval Probability", f"{ann_prob * 100:.2f}%")
            st.progress(min(max(ann_prob, 0.0), 1.0))
            if ann_prob >= threshold:
                st.success(f"DECISION: APPROVED (Score {ann_prob:.2f} >= Cutoff {threshold:.2f})")
            else:
                st.error(f"DECISION: REJECTED (Score {ann_prob:.2f} < Cutoff {threshold:.2f})")

        with c2:
            st.markdown("### Baseline Model: Random Forest")
            st.metric("Approval Probability", f"{rf_prob * 100:.2f}%")
            st.progress(min(max(rf_prob, 0.0), 1.0))
            if rf_prob >= threshold:
                st.info(f"Baseline Decision: APPROVED ({rf_prob:.2f})")
            else:
                st.warning(f"Baseline Decision: REJECTED ({rf_prob:.2f})")

with tab2:
    st.subheader("Test Set Evaluation Metrics")
    model_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, dict) and "accuracy" in v
    }
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d1e7dd"), use_container_width=True)

    if "Best_ANN_Hyperparameters" in metrics:
        st.markdown("**Selected ANN Hyperparameters:**")
        st.json(metrics["Best_ANN_Hyperparameters"])

    if os.path.exists("reports/evaluation_plots.png"):
        st.subheader("Confusion Matrix, ROC Curve, Loss Convergence & Feature Importance")
        st.image("reports/evaluation_plots.png", use_container_width=True)