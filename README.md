# Neural Network-Based Loan Approval Classification System

End-to-end Credit Underwriting Classification pipeline comparing a Traditional ML Baseline (Random Forest) against a tuned Artificial Neural Network (ANN).

## Quickstart
1. Activate environment: .\.venv\Scripts\Activate.ps1
2. Install dependencies: pip install -r requirements.txt
3. Train models & generate reports: python src/train.py
4. Launch live dashboard: streamlit run app/app.py

## Deliverables Included
- data/cleaned_loan_data.csv — Outlier-filtered & feature-engineered dataset
- notebooks/ — Full training & EDA notebook
- models/loan_ann_model.h5 — Trained Keras ANN model & serialized preprocessor
- reports/evaluation_metrics.json & reports/evaluation_plots.png — Test metrics & diagnostic plots
- app/app.py — Interactive Streamlit underwriting dashboard
- ARCHITECTURE.md — System architecture & troubleshooting guide
