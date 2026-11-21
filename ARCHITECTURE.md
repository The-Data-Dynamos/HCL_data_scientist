# System Architecture & API Contract v1.0.0 Guide

## 1. 3-Tier End-to-End Architecture

```text
[ Data Layer ]
data/loan_data.csv -> Cleaning & Outlier Filtering -> data/cleaned_loan_data.csv
                 -> Domain Feature Engineering (calc_debt_to_income, estimated_annual_interest)
                 -> ColumnTransformer Pipeline -> models/preprocessor.joblib
                 -> Keras ANN Classifier -> models/loan_ann_model.h5

[ Application Tier (FastAPI Backend) ]
backend/main.py (Port 8000)
  ├── GET  /health                                 (Service health & model readiness)
  ├── GET  /api/v1/model-info                      (Model metadata & test metrics)
  ├── POST /api/v1/predict                         (9 frontend fields -> ANN inference)
  ├── GET  /api/v1/assessments                     (Privacy-safe SQLite audit trail)
  └── GET  /api/v1/assessments/{assessment_id}     (Individual audit record lookup)
backend/database.py -> SQLite (smartloan.db)       (Strictly excludes PII/income/credit scores)

[ Presentation Tier (Streamlit Frontend) ]
frontend/app.py & app/app.py (Port 8501)
  ├── Tab 1: Loan Assessment (9 applicant fields, live DTI preview, decision card, warnings)
  ├── Tab 2: Model Performance (Dual-model comparison, ANN specs, 4-panel diagnostic plots)
  ├── Tab 3: Assessment History (Non-sensitive audit log table & ID record inspector)
  └── Sidebar Controls (Provisional Mock Mode toggle, API endpoint URL, policy cutoff slider)
```

## 2. API Contract v1.0.0 Specifications

### POST `/api/v1/predict` Request Body
```json
{
  "person_income": 65000.0,
  "person_emp_exp": 5.0,
  "person_home_ownership": "MORTGAGE",
  "loan_amnt": 10000.0,
  "loan_intent": "EDUCATION",
  "loan_int_rate": 8.5,
  "cb_person_cred_hist_length": 6.0,
  "credit_score": 720,
  "previous_loan_defaults_on_file": "No",
  "decision_threshold": 0.50
}
```

### POST `/api/v1/predict` Response JSON
```json
{
  "predicted_class": 1,
  "predicted_label": "APPROVED",
  "probability_approved": 0.8842,
  "decision_threshold": 0.50,
  "model_name": "smartloan_ann_classifier",
  "model_version": "1.0.0",
  "model_type": "Artificial Neural Network (Keras)",
  "contract_version": "1.0.0",
  "loan_intent": "EDUCATION",
  "derived_features": {
    "loan_percent_income": 0.15,
    "calc_debt_to_income": 0.1538,
    "estimated_annual_interest": 850.0
  },
  "warnings": []
}
```

## 3. Privacy by Design (SQLite Auditing)
In compliance with fair lending and financial privacy standards:
- The backend logs only non-sensitive audit records: `id`, `loan_intent`, `model_name`, `model_version`, `contract_version`, `decision_threshold`, `predicted_class`, `probability_approved`, `created_at`.
- Applicant PII, gross income, age, gender, and credit bureau scores are strictly excluded from storage in `smartloan.db`.

## 4. Dual Modeling & Diagnostic Diagnostics
- **Primary:** Artificial Neural Network (Keras Sequential with Dense, He Normal, BatchNormalization, Dropout, and Adam optimizer).
- **Baseline:** Balanced Random Forest Classifier.
- **Diagnostics:** 4-panel evaluation report in `reports/evaluation_plots.png` (Confusion Matrix, AUC-ROC Comparison, Loss Convergence, Feature Importance).

## 5. Startup Commands
- Start FastAPI Backend:
  ```powershell
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
  ```
- Start Streamlit Frontend:
  ```powershell
  streamlit run app/app.py --server.port 8501
  ```
