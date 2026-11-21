import os
import json
import joblib
import requests
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="SmartLoan AI | Credit Underwriting Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-mode polish, header spacing, and executive cards
st.markdown("""
<style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
    .decision-card {
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #334155;
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Human-readable labels & constraints mapped to exact dataset schema columns
FIELD_META = {
    "person_age": {"label": "Applicant Age (Years)", "min": 18, "max": 100, "step": 1, "is_int": True},
    "person_gender": {"label": "Gender (Audit Only)"},
    "person_education": {"label": "Highest Education Level"},
    "person_income": {"label": "Annual Gross Income ($)", "min": 1000.0, "max": 5000000.0, "step": 1000.0, "is_int": False},
    "person_emp_exp": {"label": "Employment Experience (Years)", "min": 0, "max": 80, "step": 1, "is_int": True},
    "person_home_ownership": {"label": "Home Ownership Status"},
    "loan_amnt": {"label": "Requested Loan Principal ($)", "min": 500.0, "max": 500000.0, "step": 500.0, "is_int": False},
    "loan_intent": {"label": "Loan Purpose / Intent"},
    "loan_int_rate": {"label": "Proposed Interest Rate (% APR)", "min": 1.0, "max": 40.0, "step": 0.1, "is_int": False},
    "loan_percent_income": {"label": "Loan-to-Income Ratio", "min": 0.0, "max": 5.0, "step": 0.01, "is_int": False},
    "cb_person_cred_hist_length": {"label": "Credit Bureau History (Years)", "min": 0.0, "max": 60.0, "step": 1.0, "is_int": False},
    "credit_score": {"label": "Credit Bureau Score (300–850)", "min": 300, "max": 850, "step": 5, "is_int": True},
    "previous_loan_defaults_on_file": {"label": "Prior Loan Defaults on File"}
}

PRESETS = {
    "Custom / Median Profile": None,
    "Prime Applicant (Low Risk)": {
        "person_age": 34, "person_gender": "female", "person_education": "Master",
        "person_income": 115000.0, "person_emp_exp": 10, "person_home_ownership": "MORTGAGE",
        "loan_amnt": 12000.0, "loan_intent": "EDUCATION", "loan_int_rate": 7.5,
        "cb_person_cred_hist_length": 9.0, "credit_score": 780, "previous_loan_defaults_on_file": "No"
    },
    "High-Risk Applicant (Prior Default)": {
        "person_age": 22, "person_gender": "male", "person_education": "High School",
        "person_income": 24000.0, "person_emp_exp": 1, "person_home_ownership": "RENT",
        "loan_amnt": 15000.0, "loan_intent": "PERSONAL", "loan_int_rate": 18.5,
        "cb_person_cred_hist_length": 2.0, "credit_score": 520, "previous_loan_defaults_on_file": "Yes"
    },
    "Borderline Applicant (Policy Sensitive)": {
        "person_age": 27, "person_gender": "female", "person_education": "Bachelor",
        "person_income": 52000.0, "person_emp_exp": 4, "person_home_ownership": "RENT",
        "loan_amnt": 14000.0, "loan_intent": "DEBTCONSOLIDATION", "loan_int_rate": 12.8,
        "cb_person_cred_hist_length": 4.0, "credit_score": 645, "previous_loan_defaults_on_file": "No"
    }
}

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

# Sidebar: Demo Controls & Policy Threshold
with st.sidebar:
    st.header("Underwriting Controls")
    preset_choice = st.selectbox("Quick Demo Applicant Preset", list(PRESETS.keys()))
    preset_values = PRESETS[preset_choice]

    st.divider()
    threshold = st.slider(
        "Approval Probability Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Raise cutoff to tighten credit risk; lower cutoff to expand loan approvals."
    )
    st.caption(f"Current Risk Stance: **{'Conservative' if threshold > 0.55 else 'Balanced' if threshold >= 0.45 else 'Growth / Aggressive'}**")

    st.divider()
    inference_mode = st.radio(
        "Inference Engine",
        ["Local Artifacts (Direct)", "FastAPI Endpoint (REST)"],
        index=0
    )
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000", disabled=(inference_mode == "Local Artifacts (Direct)"))

    st.divider()
    st.markdown("**System Specs**")
    st.caption("• Dataset: 45,000 Credit Records\n• Split: 70% Train / 15% Val / 15% Test\n• Primary: Keras ANN (Adam + Dropout)\n• Baseline: Balanced Random Forest")

# Header Banner
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">SmartLoan AI — Credit Underwriting & Risk Classification</p>
    <p class="hero-sub">Neural Network Decision Engine (ANN) vs. Random Forest Baseline | Real-Time Policy Simulation & Diagnostics</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Live Loan Underwriting Simulator", "Model Benchmarks & Diagnostics"])

with tab1:
    input_data = {}
    cat_cols = schema.get("cat_cols", [])
    raw_num_cols = schema.get("raw_num_cols", [])

    st.subheader("1. Applicant Demographics & Credit Profile")
    c_demo = st.columns(3)
    idx = 0
    for col_name in cat_cols:
        meta = FIELD_META.get(col_name, {"label": col_name})
        options = schema["cat_values"][col_name]
        default_idx = 0
        if preset_values and col_name in preset_values and preset_values[col_name] in options:
            default_idx = options.index(preset_values[col_name])
        with c_demo[idx % 3]:
            input_data[col_name] = st.selectbox(meta["label"], options, index=default_idx, key=f"in_{col_name}")
        idx += 1

    st.subheader("2. Income, Loan Terms & Credit Bureau Metrics")
    c_num = st.columns(3)
    num_idx = 0
    for col_name in raw_num_cols:
        # Auto-derive loan_percent_income dynamically below so user inputs stay consistent
        if col_name == "loan_percent_income":
            continue
        meta = FIELD_META.get(col_name, {"label": col_name, "is_int": False})
        raw_default = preset_values[col_name] if (preset_values and col_name in preset_values) else schema["num_medians"][col_name]

        with c_num[num_idx % 3]:
            if meta.get("is_int", False):
                input_data[col_name] = st.number_input(
                    meta["label"],
                    min_value=int(meta.get("min", 0)),
                    max_value=int(meta.get("max", 10000000)),
                    value=int(round(float(raw_default))),
                    step=int(meta.get("step", 1)),
                    key=f"in_{col_name}"
                )
            else:
                input_data[col_name] = st.number_input(
                    meta["label"],
                    min_value=float(meta.get("min", 0.0)),
                    max_value=float(meta.get("max", 10000000.0)),
                    value=float(raw_default),
                    step=float(meta.get("step", 1.0)),
                    key=f"in_{col_name}"
                )
        num_idx += 1

    # Compute derived financial ratios live for display & model input
    income_val = max(float(input_data.get("person_income", 1.0)), 1.0)
    loan_val = float(input_data.get("loan_amnt", 0.0))
    rate_val = float(input_data.get("loan_int_rate", 0.0))

    dti_ratio = loan_val / income_val
    annual_interest = loan_val * (rate_val / 100.0)
    if "loan_percent_income" in raw_num_cols:
        input_data["loan_percent_income"] = round(dti_ratio, 4)

    # Live Financial Summary Strip
    st.markdown("#### Live Engineered Financial Indicators")
    m1, m2, m3 = st.columns(3)
    m1.metric("Debt-to-Income (DTI) Ratio", f"{dti_ratio * 100:.2f}%", delta="High Burden" if dti_ratio > 0.35 else "Healthy", delta_color="inverse")
    m2.metric("Estimated Annual Interest", f"${annual_interest:,.2f}")
    m3.metric("Active Approval Cutoff", f"{threshold * 100:.0f}%")

    st.divider()
    if st.button("Run Underwriting Evaluation", type="primary", use_container_width=True):
        df_in = pd.DataFrame([input_data])

        # Replicate domain feature engineering from src/train.py
        if "loan_amnt" in df_in.columns and "person_income" in df_in.columns:
            df_in["calc_debt_to_income"] = df_in["loan_amnt"] / (df_in["person_income"] + 1.0)
        if "loan_amnt" in df_in.columns and "loan_int_rate" in df_in.columns:
            df_in["estimated_annual_interest"] = df_in["loan_amnt"] * (df_in["loan_int_rate"] / 100.0)

        ann_prob, rf_prob = None, None

        if inference_mode == "FastAPI Endpoint (REST)":
            try:
                api_payload = {k: v for k, v in input_data.items() if k not in ["person_gender", "loan_percent_income"]}
                api_payload["decision_threshold"] = float(threshold)
                resp = requests.post(f"{api_url}/api/v1/predict", json=api_payload, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    ann_prob = float(data["approval_probability"])
                    rf_prob = float(data["baseline_probability"])
                else:
                    st.error(f"FastAPI returned HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                st.warning(f"FastAPI unreachable ({e}). Falling back to local serialized models.")

        if ann_prob is None:
            X_proc = preprocessor.transform(df_in)
            ann_prob = float(ann_model.predict(X_proc, verbose=0)[0][0])
            rf_prob = float(rf_model.predict_proba(X_proc)[0][1])

        st.subheader("Underwriting Decision Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Primary Engine: Artificial Neural Network")
            st.metric("Predicted Approval Probability", f"{ann_prob * 100:.2f}%", delta=f"{(ann_prob - threshold) * 100:+.2f}% vs Cutoff")
            st.progress(min(max(ann_prob, 0.0), 1.0))
            if ann_prob >= threshold:
                st.success(f"FINAL DECISION: APPROVED (Probability {ann_prob:.2f} >= Cutoff {threshold:.2f})")
            else:
                st.error(f"FINAL DECISION: REJECTED (Probability {ann_prob:.2f} < Cutoff {threshold:.2f})")

        with c2:
            st.markdown("### Benchmark Engine: Random Forest")
            st.metric("Predicted Approval Probability", f"{rf_prob * 100:.2f}%", delta=f"{(rf_prob - threshold) * 100:+.2f}% vs Cutoff")
            st.progress(min(max(rf_prob, 0.0), 1.0))
            if rf_prob >= threshold:
                st.info(f"BASELINE DECISION: APPROVED (Probability {rf_prob:.2f} >= Cutoff {threshold:.2f})")
            else:
                st.warning(f"BASELINE DECISION: REJECTED (Probability {rf_prob:.2f} < Cutoff {threshold:.2f})")

        # Risk & Consensus Summary
        st.markdown("#### Automated Underwriting Audit Notes")
        reasons = []
        if input_data.get("previous_loan_defaults_on_file") == "Yes":
            reasons.append("• **High Risk Flag:** Applicant has a prior loan default on credit bureau file.")
        if dti_ratio > 0.35:
            reasons.append(f"• **Elevated Debt Burden:** Requested loan represents **{dti_ratio*100:.1f}%** of annual income (>35% threshold).")
        if input_data.get("credit_score", 650) >= 720:
            reasons.append(f"• **Strong Credit Tier:** Credit score of **{input_data.get('credit_score')}** indicates prime repayment history.")
        elif input_data.get("credit_score", 650) < 580:
            reasons.append(f"• **Subprime Credit Tier:** Credit score of **{input_data.get('credit_score')}** falls below standard underwriting floor.")

        consensus = (ann_prob >= threshold) == (rf_prob >= threshold)
        reasons.append(
            "• **Model Consensus:** Both the Artificial Neural Network and Random Forest baseline agree on this decision."
            if consensus else
            "• **Model Divergence:** ANN and Random Forest disagree near the decision boundary — recommend manual underwriter review."
        )
        for r in reasons:
            st.markdown(r)

with tab2:
    st.subheader("Unseen Test Set Evaluation Benchmarks (15% Stratified Split)")
    model_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, dict) and "accuracy" in v
    }
    if model_metrics:
        metrics_df = pd.DataFrame(model_metrics).T
        # Dark-mode safe high-contrast styling
        styled_df = (
            metrics_df.style
            .format("{:.4f}")
            .highlight_max(axis=0, color="#14532d")
            .set_properties(**{"color": "#f8fafc", "font-weight": "500"})
        )
        st.dataframe(styled_df, use_container_width=True)

    col_hp, col_info = st.columns([1, 2])
    with col_hp:
        if "Best_ANN_Hyperparameters" in metrics:
            st.markdown("**Selected ANN Hyperparameters**")
            st.json(metrics["Best_ANN_Hyperparameters"])
    with col_info:
        st.markdown("**Architecture & Regularization Summary**")
        st.info(
            "• **Input & Preprocessing:** Median/Mode Imputation + StandardScaler + OneHotEncoder (fit strictly on X_train).\n"
            "• **Neural Network:** 2 Hidden Dense Layers (ReLU + He Normal Init) + BatchNormalization + Dropout.\n"
            "• **Optimization:** Adam Optimizer + Binary Cross-Entropy + Balanced Class Weights + Early Stopping."
        )

    if os.path.exists("reports/evaluation_plots.png"):
        st.subheader("Diagnostic Visuals: Confusion Matrix, ROC Curve, Loss Convergence & Feature Importance")
        st.image("reports/evaluation_plots.png", use_container_width=True)