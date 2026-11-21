import os
import sys
import json
import joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

# Ensure required project directories exist
for folder in ["data", "models", "reports", "notebooks"]:
    os.makedirs(folder, exist_ok=True)

# 1. Data Ingestion & Cleaning
RAW_PATH = "data/loan_data.csv"
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Missing {RAW_PATH}. Ensure loan_data.csv is placed in data/.")

df = pd.read_csv(RAW_PATH)
df = df.drop_duplicates().copy()

# Filter unrealistic age outliers (valid adult range: 18–100)
if "person_age" in df.columns:
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)].copy()

# Validate employment experience against age
if "person_emp_exp" in df.columns and "person_age" in df.columns:
    df = df[(df["person_emp_exp"] >= 0) & (df["person_emp_exp"] <= (df["person_age"] - 18))].copy()

TARGET_COL = "loan_status"
if df[TARGET_COL].dtype == "object" or str(df[TARGET_COL].dtype) == "string":
    df[TARGET_COL] = df[TARGET_COL].str.strip().map({
        "Y": 1, "Yes": 1, "Approved": 1, "1": 1,
        "N": 0, "No": 0, "Rejected": 0, "0": 0
    })
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Retain all dataset features (including person_gender)
df_model = df.copy()

# Domain Feature Engineering
if "loan_amnt" in df_model.columns and "person_income" in df_model.columns:
    df_model["calc_debt_to_income"] = df_model["loan_amnt"] / (df_model["person_income"] + 1.0)
if "loan_amnt" in df_model.columns and "loan_int_rate" in df_model.columns:
    df_model["estimated_annual_interest"] = df_model["loan_amnt"] * (df_model["loan_int_rate"] / 100.0)

# Save cleaned dataset deliverable
df_model.to_csv("data/cleaned_loan_data.csv", index=False)
print(f"Saved cleaned dataset ({len(df_model)} rows) -> data/cleaned_loan_data.csv")

X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL]

# Pandas 2 & 3/4 compatible dtype selection
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

engineered_cols = ["calc_debt_to_income", "estimated_annual_interest"]
raw_num_cols = [c for c in num_cols if c not in engineered_cols]

schema = {
    "raw_num_cols": raw_num_cols,
    "cat_cols": cat_cols,
    "all_num_cols": num_cols,
    "cat_values": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols},
    "num_medians": {c: float(X[c].median()) for c in raw_num_cols},
    "num_mins": {c: float(X[c].min()) for c in raw_num_cols},
    "num_maxs": {c: float(X[c].max()) for c in raw_num_cols}
}
with open("models/feature_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

# 2. Stratified Train / Validation / Test Split (70% / 15% / 15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 3. Leakage-Free Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)
joblib.dump(preprocessor, "models/preprocessor.joblib")

# Extract transformed feature names for Feature Importance visualization
ohe_features = (
    preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(cat_cols).tolist()
    if cat_cols else []
)
feature_names = num_cols + ohe_features

# Compute balanced class weights for class imbalance
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# 4A. Traditional ML Baseline (Random Forest)
print("Training Random Forest baseline...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42
)
rf_model.fit(X_train_proc, y_train)
joblib.dump(rf_model, "models/rf_baseline.joblib")

rf_preds = rf_model.predict(X_test_proc)
rf_probs = rf_model.predict_proba(X_test_proc)[:, 1]

# 4B. Neural Network Builder & Validation Hyperparameter Tuning
def build_ann(input_dim, units_1=64, units_2=32, dropout_rate=0.30, lr=0.001):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units_1, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units_2, activation="relu", kernel_initializer="he_normal"),
        BatchNormalization(),
        Dropout(round(dropout_rate * 0.67, 2)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

param_grid = [
    {"units_1": 64, "units_2": 32, "dropout_rate": 0.30, "lr": 0.001},
    {"units_1": 128, "units_2": 64, "dropout_rate": 0.25, "lr": 0.0005}
]

best_val_auc = -1.0
best_model = None
best_history = None
best_params = None

tf.random.set_seed(42)
for params in param_grid:
    print(f"Tuning ANN with params: {params}")
    candidate = build_ann(X_train_proc.shape[1], **params)
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    hist = candidate.fit(
        X_train_proc, y_train,
        validation_data=(X_val_proc, y_val),
        epochs=40,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0
    )
    val_auc = max(hist.history["val_auc"])
    print(f"  -> Best Validation AUC: {val_auc:.4f}")
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model = candidate
        best_history = hist
        best_params = params

print(f"\nSelected Best ANN Hyperparameters: {best_params}")

# Save both modern .keras and legacy .h5 artifacts
best_model.save("models/smartloan_ann_v1.keras")
best_model.save("models/loan_ann_model.h5")

ann_probs = best_model.predict(X_test_proc, verbose=0).ravel()
ann_preds = (ann_probs >= 0.50).astype(int)

# 5. Evaluation Metrics & Model Metadata Contract Export
def get_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4)
    }

report = {
    "Best_ANN_Hyperparameters": best_params,
    "Random_Forest_Baseline": get_metrics(y_test, rf_preds, rf_probs),
    "Artificial_Neural_Network": get_metrics(y_test, ann_preds, ann_probs)
}
with open("reports/evaluation_metrics.json", "w") as f:
    json.dump(report, f, indent=2)

metadata = {
    "model_name": "smartloan_ann_classifier",
    "model_version": "1.0.0",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_rows_cleaned": int(len(df_model)),
    "features_used": {
        "categorical": cat_cols,
        "numeric_raw": raw_num_cols,
        "numeric_derived": engineered_cols
    },
    "class_distribution": {
        "rejected_0": int((y == 0).sum()),
        "approved_1": int((y == 1).sum())
    },
    "class_label_mapping": {"0": "REJECTED", "1": "APPROVED"},
    "default_threshold": 0.50,
    "supported_loan_categories": schema["cat_values"].get("loan_intent", []),
    "environment_versions": {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__
    }
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=== Test Set Evaluation Report ===")
print(json.dumps(report, indent=2))

# 6. Visual Outputs: Confusion Matrix, ROC Curve, Loss Curve & Feature Importance
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel 1: Confusion Matrix
cm = confusion_matrix(y_test, ann_preds)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
    xticklabels=["Rejected (0)", "Approved (1)"],
    yticklabels=["Rejected (0)", "Approved (1)"]
)
axes[0, 0].set_title("ANN Confusion Matrix (Test Set)", fontweight="bold")
axes[0, 0].set_xlabel("Predicted Label")
axes[0, 0].set_ylabel("Actual Label")

# Panel 2: ROC Curve Comparison
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_ann, tpr_ann, _ = roc_curve(y_test, ann_probs)
axes[0, 1].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {report['Random_Forest_Baseline']['auc_roc']})")
axes[0, 1].plot(fpr_ann, tpr_ann, label=f"Neural Network ANN (AUC = {report['Artificial_Neural_Network']['auc_roc']})", linewidth=2.2)
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0, 1].set_title("AUC-ROC Curve Comparison", fontweight="bold")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(loc="lower right")

# Panel 3: Training vs Validation Loss
axes[1, 0].plot(best_history.history["loss"], label="Train Loss", linewidth=2)
axes[1, 0].plot(best_history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[1, 0].set_title("ANN Loss Convergence (Early Stopping)", fontweight="bold")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("Binary Cross-Entropy Loss")
axes[1, 0].legend()

# Panel 4: Top 10 Driving Features
importances = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(10).sort_values()
importances.plot(kind="barh", color="#1e40af", ax=axes[1, 1])
axes[1, 1].set_title("Top 10 Driving Features (Feature Importance)", fontweight="bold")
axes[1, 1].set_xlabel("Relative Importance")

plt.tight_layout()
plt.savefig("reports/evaluation_plots.png", dpi=150)
print("Saved 4-panel visual report -> reports/evaluation_plots.png")