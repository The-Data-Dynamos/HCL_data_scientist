# SmartLoan AI — Neural Network-Based Loan Approval Classification System

End-to-end credit underwriting classification system built for the **HCLTech Hackathon** (`The-Data-Dynamos/HCL_data_scientist`). SmartLoan AI combines a leakage-free machine learning pipeline, a versioned FastAPI inference service, and an interactive Streamlit underwriting dashboard adhering to **API Contract v1.0.0**.

> **Disclaimer:** SmartLoan AI is a hackathon demonstration system trained on synthetic credit records. Model scores represent experimental underwriting assessments—not guaranteed lending decisions or calibrated real-world approval probabilities.

---

## System Architecture

```text
Raw Dataset (data/loan_data.csv)
   │
   ▼
ML Pipeline (src/train.py & notebooks/)
   ├── Cleaning, Outlier Filtering & Domain Feature Engineering
   ├── Stratified 70% Train / 15% Validation / 15% Test Split
   ├── Preprocessing Pipeline (SimpleImputer + StandardScaler + OneHotEncoder)
   └── Dual Modeling: Balanced Random Forest Baseline vs. Tuned Keras ANN
   │
   ▼
Serialized Artifacts (models/ & reports/)
   ├── loan_ann_model.h5 / smartloan_ann_v1.keras
   ├── rf_baseline.joblib & preprocessor.joblib
   ├── feature_schema.json & metadata.json
   └── evaluation_metrics.json & evaluation_plots.png
   │
   ▼
FastAPI Inference Backend ([http://127.0.0.1:8000](http://127.0.0.1:8000))
   ├── GET  /health
   ├── GET  /api/v1/model-info
   ├── POST /api/v1/predict
   └── GET  /api/v1/assessments
   │
   ▼
Streamlit Underwriting Portal (app/app.py — http://localhost:8501)
   ├── 1. Loan Assessment (9-field contract form, ₹ currency labels, provisional mock toggle)
   ├── 2. Model Performance (benchmark metrics, confusion matrix, ROC curve)
   └── 3. Assessment History (saved audit logs & detail inspector)
