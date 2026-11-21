import os
import pandas as pd
import streamlit as st

from services.api_client import SmartLoanAPIClient

# Configure page layout and visual identity
st.set_page_config(
    page_title="SmartLoan  | Credit Underwriting Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-mode styling, spacing, and card layouts
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Applicant Presets for demo and rapid evaluation
PRESETS = {
    "Custom / Fresh Form": None,
    "Prime Applicant (Low Risk)": {
        "person_income": 115000.0,
        "person_emp_exp": 10.0,
        "person_home_ownership": "MORTGAGE",
        "loan_amnt": 12000.0,
        "loan_intent": "EDUCATION",
        "loan_int_rate": 7.5,
        "cb_person_cred_hist_length": 9.0,
        "credit_score": 780,
        "previous_loan_defaults_on_file": "No"
    },
    "High-Risk Applicant (Prior Default)": {
        "person_income": 24000.0,
        "person_emp_exp": 1.0,
        "person_home_ownership": "RENT",
        "loan_amnt": 15000.0,
        "loan_intent": "PERSONAL",
        "loan_int_rate": 18.5,
        "cb_person_cred_hist_length": 2.0,
        "credit_score": 520,
        "previous_loan_defaults_on_file": "Yes"
    },
    "Borderline Applicant (Policy Sensitive)": {
        "person_income": 52000.0,
        "person_emp_exp": 4.0,
        "person_home_ownership": "RENT",
        "loan_amnt": 14000.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_int_rate": 12.8,
        "cb_person_cred_hist_length": 4.0,
        "credit_score": 645,
        "previous_loan_defaults_on_file": "No"
    }
}

# Backend client configuration (Zero direct ML model inference in Streamlit)
api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
mock_mode = False
client = SmartLoanAPIClient(base_url=api_url, mock_mode=mock_mode)

# Sidebar: Presets & Policy Threshold Controls
with st.sidebar:
    # Preset Selection
    preset_choice = st.selectbox("Applicant Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_choice]

    st.divider()

    # Policy Cutoff Threshold Slider
    threshold = st.slider(
        "Approval Cutoff Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Raise cutoff to tighten underwriting policy; lower cutoff to expand credit issuance."
    )
    stance = "Conservative (Low Risk)" if threshold > 0.55 else "Balanced (Standard)" if threshold >= 0.45 else "Growth (High Approval)"
    st.caption(f"Risk Policy Stance: **{stance}**")

    st.divider()
    st.markdown("**API Contract v1.0.0 Architecture**")
    st.caption(
        "• **Frontend:** Streamlit 3-Tier Client\n"
        "• **Client Adapter:** `services.api_client`\n"
        "• **Backend:** FastAPI REST Engine (`:8000`)\n"
        "• **Auditing:** SQLite Non-Sensitive DB (`smartloan.db`)\n"
        "• **Inference:** Tuned Keras ANN"
    )

# Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">SmartLoan AI — Credit Underwriting Engine</p>
    <p class="hero-sub">Production 3-Tier Credit Decisioning: Streamlit Frontend → FastAPI Backend → Keras Neural Network</p>
</div>
""", unsafe_allow_html=True)

# 3 Production Tabs per Contract v1.0.0 Specification
tab1, tab2, tab3 = st.tabs([
    "1. Loan Assessment",
    "2. Model Performance",
    "3. Assessment History"
])

# ==============================================================================
# TAB 1: LOAN ASSESSMENT (9 Required Frontend Fields)
# ==============================================================================
with tab1:
    st.subheader("Applicant Underwriting Assessment Form")
    st.caption("All 9 required fields strictly validated against Contract v1.0.0.")

    with st.form("loan_assessment_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            person_income = st.number_input(
                "Gross Annual Income (₹)",
                min_value=0.0,
                max_value=10_000_000.0,
                value=float(preset["person_income"]) if preset else 25000.0,
                step=1000.0,
            )
            person_emp_exp = st.number_input(
                "Employment Experience (Years)",
                min_value=0.0,
                max_value=60.0,
                value=float(preset["person_emp_exp"]) if preset else 5.0,
                step=1.0
            )
            home_options = ["MORTGAGE", "RENT", "OWN", "OTHER"]
            home_idx = home_options.index(preset["person_home_ownership"]) if preset and preset["person_home_ownership"] in home_options else 0
            person_home_ownership = st.selectbox("Home Ownership Status", home_options, index=home_idx)

        with col_b:
            loan_amnt = st.number_input(
                "Requested Loan Principal (₹)",
                min_value=0.0,
                max_value=35_000.0,
                value=float(preset["loan_amnt"]) if preset else 9000.0,
                step=500.0,
            )
            intent_options = ["EDUCATION", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
            intent_idx = intent_options.index(preset["loan_intent"]) if preset and preset["loan_intent"] in intent_options else 0
            loan_intent = st.selectbox("Loan Purpose / Intent", intent_options, index=intent_idx)

            loan_int_rate = st.number_input(
                "Proposed Interest Rate (% APR)",
                min_value=1.0,
                max_value=40.0,
                value=float(preset["loan_int_rate"]) if preset else 10.5,
                step=0.1
            )

        with col_c:
            cb_person_cred_hist_length = st.number_input(
                "Credit Bureau History (Years)",
                min_value=0.0,
                max_value=60.0,
                value=float(preset["cb_person_cred_hist_length"]) if preset else 5.0,
                step=1.0
            )
            credit_score = st.number_input(
                "Credit Bureau Score (300–850)",
                min_value=300,
                max_value=850,
                value=int(preset["credit_score"]) if preset else 680,
                step=5
            )
            default_options = ["No", "Yes"]
            default_idx = default_options.index(preset["previous_loan_defaults_on_file"]) if preset and preset["previous_loan_defaults_on_file"] in default_options else 0
            previous_loan_defaults_on_file = st.selectbox("Prior Loan Defaults on File", default_options, index=default_idx)

        # Real-time UI indicator preview
        dti_preview = round(loan_amnt / max(person_income, 1.0), 2)
        annual_interest_preview = round(loan_amnt * (loan_int_rate / 100.0), 2)

        st.markdown("##### Live Form Indicators")
        i1, i2, i3 = st.columns(3)
        i1.metric("Calculated Loan-to-Income", f"{dti_preview * 100:.1f}%", delta="High Burden" if dti_preview > 0.40 else "Healthy", delta_color="inverse")
        i2.metric("Estimated Annual Interest", f"₹{annual_interest_preview:,.2f}")
        i3.metric("Decision Cutoff Applied", f"{threshold * 100:.0f}%")

        submit_btn = st.form_submit_button("Run Underwriting Assessment", type="primary", use_container_width=True)

    if submit_btn:
        payload = {
            "person_income": float(person_income),
            "person_emp_exp": float(person_emp_exp),
            "person_home_ownership": person_home_ownership,
            "loan_amnt": float(loan_amnt),
            "loan_intent": loan_intent,
            "loan_int_rate": float(loan_int_rate),
            "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
            "credit_score": int(credit_score),
            "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
            "decision_threshold": float(threshold)
        }

        with st.spinner("Submitting application to Underwriting API..."):
            try:
                # Zero direct ML inference in Streamlit: delegate solely to FastAPI client
                result = client.predict_loan(payload)

                st.divider()
                st.subheader("Official Underwriting Decision")

                # Decision Header
                is_approved = result["predicted_class"] == 1
                if is_approved:
                    st.success(f"### ✅ LOAN APPROVED — {result['predicted_label']}")
                else:
                    st.error(f"### ❌ LOAN REJECTED — {result['predicted_label']}")

                # Primary Result Columns
                r_prob, r_metrics, r_contract = st.columns([1.2, 1.2, 1.2])

                with r_prob:
                    prob = result["probability_approved"]
                    delta_pct = (prob - threshold) * 100
                    st.metric(
                        "Approval Probability",
                        f"{prob * 100:.2f}%",
                        delta=f"{delta_pct:+.2f}% vs Threshold ({threshold * 100:.0f}%)"
                    )
                    st.progress(min(max(prob, 0.0), 1.0))

                with r_metrics:
                    st.markdown("**Engineered Ratios**")
                    derived = result.get("derived_features", {})
                    st.write(f"• **Loan-to-Income:** `{derived.get('loan_percent_income', 0.0) * 100:.1f}%`")
                    st.write(f"• **Debt-to-Income:** `{derived.get('calc_debt_to_income', 0.0):.4f}`")
                    st.write(f"• **Est. Annual Interest:** `₹{derived.get('estimated_annual_interest', 0.0):,.2f}`")

                with r_contract:
                    st.markdown("**Contract & Governance Specs**")
                    st.write(f"• **Model:** `{result.get('model_name')}`")
                    st.write(f"• **Version:** `{result.get('model_version')}` (v{result.get('contract_version')})")
                    st.write(f"• **Type:** `{result.get('model_type')}`")
                    st.write(f"• **Intent:** `{result.get('loan_intent')}`")

                # Warnings & Policy Audit Flags
                warnings = result.get("warnings", [])
                st.markdown("#### Automated Risk & Policy Flags")
                if warnings:
                    for w in warnings:
                        st.warning(f"⚠️ {w}")
                else:
                    st.info("✅ All risk indicators satisfy standard underwriting policy guidelines.")

            except Exception as e:
                st.error(f"Underwriting API error: {e}")
                st.info("💡 Tip: Make sure the FastAPI server is running on port 8000 (`uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload`).")


# ==============================================================================
# TAB 2: MODEL PERFORMANCE (Contract v1.0.0 Benchmarks & Diagnostics)
# ==============================================================================
with tab2:
    st.subheader("Model Performance & Diagnostic Benchmarks")
    st.caption("Contract v1.0.0 Evaluation on Unseen 15% Stratified Test Partition.")

    try:
        model_info = client.get_model_info()

        # Display System Architecture Overview
        c_spec1, c_spec2 = st.columns([1, 2])
        with c_spec1:
            st.markdown("##### Production Model Specs")
            st.markdown(f"**Model Identifier:** `{model_info.get('model_name')}`")
            st.markdown(f"**Version:** `{model_info.get('model_version')}`")
            st.markdown(f"**Algorithm:** `{model_info.get('model_type')}`")
            st.markdown(f"**Contract:** `v{model_info.get('contract_version')}`")
            st.markdown(f"**Default Policy Cutoff:** `{model_info.get('default_threshold', 0.50):.2f}`")

        with c_spec2:
            st.markdown("##### Features Used by Inference Pipeline")
            feat_breakdown = model_info.get("features_used", {})
            st.write(f"• **Categorical Features:** {', '.join(feat_breakdown.get('categorical', []))}")
            st.write(f"• **Raw Numeric Features:** {', '.join(feat_breakdown.get('numeric_raw', []))}")
            st.write(f"• **Engineered Features:** {', '.join(feat_breakdown.get('numeric_derived', []))}")

        st.divider()

        # Model Benchmark Comparison Table
        eval_metrics = model_info.get("evaluation_metrics", {})
        benchmark_data = {}
        for k in ["Random_Forest_Baseline", "Artificial_Neural_Network"]:
            if k in eval_metrics:
                benchmark_data[k.replace("_", " ")] = eval_metrics[k]

        if benchmark_data:
            st.markdown("##### Dual-Model Evaluation Metrics (Test Set)")
            df_metrics = pd.DataFrame(benchmark_data).T
            styled_metrics = (
                df_metrics.style
                .format("{:.4f}")
                .highlight_max(axis=0, color="#14532d")
                .set_properties(**{"font-weight": "600", "color": "#f8fafc"})
            )
            st.dataframe(styled_metrics, use_container_width=True)

        # ANN Hyperparameters
        if "Best_ANN_Hyperparameters" in eval_metrics:
            st.markdown("##### Tuned ANN Hyperparameters")
            st.json(eval_metrics["Best_ANN_Hyperparameters"])

        # Diagnostic Visualizations
        plots_path = "reports/evaluation_plots.png"
        if os.path.exists(plots_path):
            st.divider()
            st.markdown("##### Diagnostic Plots: Confusion Matrix, ROC Comparison, Loss Convergence & Feature Importance")
            st.image(plots_path, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to fetch model info: {e}")


# ==============================================================================
# TAB 3: ASSESSMENT HISTORY (Privacy-Safe SQLite Audit Log)
# ==============================================================================
with tab3:
    st.subheader("Underwriting Assessment History & Audit Trail")
    st.info(
        "🛡️ **Privacy by Design:** In strict adherence to financial privacy regulations, "
        "no applicant PII, gross income, age, gender, or credit scores are persisted in this audit log."
    )

    h_col1, h_col2 = st.columns([3, 1])
    with h_col2:
        refresh_btn = st.button("🔄 Refresh Audit Log", use_container_width=True)

    try:
        assessments = client.get_assessments(limit=100)

        if assessments:
            df_hist = pd.DataFrame(assessments)
            # Reorder for clarity
            display_cols = [
                "id", "loan_intent", "predicted_class", "probability_approved",
                "decision_threshold", "model_name", "model_version", "created_at"
            ]
            valid_cols = [c for c in display_cols if c in df_hist.columns]
            df_hist = df_hist[valid_cols]

            st.dataframe(
                df_hist,
                use_container_width=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", format="%d"),
                    "predicted_class": st.column_config.NumberColumn("Class (1=Appr, 0=Rej)", format="%d"),
                    "probability_approved": st.column_config.ProgressColumn("Approval Probability", min_value=0.0, max_value=1.0, format="%.2f"),
                    "decision_threshold": st.column_config.NumberColumn("Cutoff", format="%.2f"),
                    "created_at": st.column_config.DatetimeColumn("Timestamp (UTC)")
                }
            )
            st.caption(f"Displaying {len(assessments)} recent non-sensitive audit records.")
        else:
            st.info("No assessment records found in the audit trail yet. Submit an application in Tab 1!")

        st.divider()

        # Single Record Lookup Tool
        st.markdown("##### Audit Record Lookup")
        lookup_col1, lookup_col2 = st.columns([2, 1])
        with lookup_col1:
            lookup_id = st.number_input("Enter Assessment ID", min_value=1, step=1, value=1)
        with lookup_col2:
            st.write("")
            st.write("")
            lookup_btn = st.button("Lookup Record", use_container_width=True)

        if lookup_btn:
            rec = client.get_assessment_by_id(int(lookup_id))
            if rec:
                st.success(f"Audit Record #{lookup_id} Found:")
                st.json(rec)
            else:
                st.warning(f"No assessment record found with ID #{lookup_id}.")

    except Exception as e:
        st.error(f"Failed to fetch assessment history: {e}")