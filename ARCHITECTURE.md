# System Architecture & Troubleshooting Guide

## 1. End-to-End Architecture
Raw CSV (loan_data.csv) -> Cleaning & Domain Feature Engineering -> Stratified Split (70/15/15) -> ColumnTransformer (StandardScaler + OneHotEncoder) -> Dual Modeling (Random Forest Baseline + Tuned Keras ANN) -> Evaluation (JSON + 4-Panel Diagnostics) -> Streamlit Inference UI

## 2. Neural Network Design Choices
- **Architecture:** Input -> Dense(64/128, ReLU, He Normal) -> BatchNormalization -> Dropout -> Dense(32/64, ReLU, He Normal) -> BatchNormalization -> Dropout -> Dense(1, Sigmoid)
- **Loss & Optimization:** Binary Crossentropy loss with Adam optimizer and balanced class weights to handle approval/rejection imbalance.
- **Regularization:** Dropout layers combined with Early Stopping (patience=6 on val_loss) to prevent overfitting.

## 3. Troubleshooting
- **Missing Artifacts Error in UI:** Run python src/train.py before launching streamlit run app/app.py.
- **Port Conflict:** Run streamlit run app/app.py --server.port 8502 if port 8501 is in use.
