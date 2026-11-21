import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import tensorflow as tf

from backend.schemas import PredictionRequest, PredictionResponse, DerivedFeatures

# Suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ModelService:
    def __init__(self, models_dir: str = "models", reports_dir: str = "reports"):
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.preprocessor = None
        self.ann_model = None
        self.schema: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.is_loaded = False

    def load_artifacts(self) -> None:
        """Load serialized preprocessor, ANN model, schema, and metrics."""
        preprocessor_path = os.path.join(self.models_dir, "preprocessor.joblib")
        model_path = os.path.join(self.models_dir, "loan_ann_model.h5")
        schema_path = os.path.join(self.models_dir, "feature_schema.json")
        metadata_path = os.path.join(self.models_dir, "metadata.json")
        metrics_path = os.path.join(self.reports_dir, "evaluation_metrics.json")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Missing preprocessor at {preprocessor_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model at {model_path}")

        self.preprocessor = joblib.load(preprocessor_path)
        self.ann_model = tf.keras.models.load_model(model_path)

        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        self.is_loaded = True

    def predict(self, req: PredictionRequest) -> Tuple[PredictionResponse, Dict[str, Any]]:
        """
        Adapts the 9 validated frontend inputs to the serialized preprocessor pipeline,
        computes derived features, performs ANN inference, and generates risk warnings.
        """
        if not self.is_loaded:
            self.load_artifacts()

        # 1. Compute derived financial features per Contract v1.0.0
        loan_percent_income = round(float(req.loan_amnt) / float(req.person_income), 2)
        calc_debt_to_income = float(req.loan_amnt) / (float(req.person_income) + 1.0)
        estimated_annual_interest = float(req.loan_amnt) * (float(req.loan_int_rate) / 100.0)

        derived = DerivedFeatures(
            loan_percent_income=loan_percent_income,
            calc_debt_to_income=round(calc_debt_to_income, 4),
            estimated_annual_interest=round(estimated_annual_interest, 2)
        )

        # 2. Preprocessor adapter: supply schema-trained neutral baselines for non-PII demographics
        medians = self.schema.get("num_medians", {})
        default_age = float(medians.get("person_age", 26.0))
        default_gender = "female"
        default_education = "Bachelor"

        # Construct input row matching preprocessor feature expectations
        row_data = {
            "person_age": default_age,
            "person_gender": default_gender,
            "person_education": default_education,
            "person_income": float(req.person_income),
            "person_emp_exp": float(req.person_emp_exp),
            "person_home_ownership": req.person_home_ownership,
            "loan_amnt": float(req.loan_amnt),
            "loan_intent": req.loan_intent,
            "loan_int_rate": float(req.loan_int_rate),
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": float(req.cb_person_cred_hist_length),
            "credit_score": float(req.credit_score),
            "previous_loan_defaults_on_file": req.previous_loan_defaults_on_file,
            "calc_debt_to_income": calc_debt_to_income,
            "estimated_annual_interest": estimated_annual_interest
        }

        df_in = pd.DataFrame([row_data])

        # Align column order if preprocessor defines feature_names_in_
        if hasattr(self.preprocessor, "feature_names_in_"):
            expected_cols = list(self.preprocessor.feature_names_in_)
            for col in expected_cols:
                if col not in df_in.columns:
                    df_in[col] = medians.get(col, 0.0)
            df_in = df_in[expected_cols]

        # 3. Leakage-free transformation and ANN inference
        X_proc = self.preprocessor.transform(df_in)
        raw_prob = float(self.ann_model.predict(X_proc, verbose=0)[0][0])
        probability_approved = round(raw_prob, 4)

        # 4. Underwriting threshold decision
        threshold = float(req.decision_threshold if req.decision_threshold is not None else 0.50)
        predicted_class = 1 if probability_approved >= threshold else 0
        predicted_label = "APPROVED" if predicted_class == 1 else "REJECTED"

        # 5. Risk and policy warnings
        warnings: List[str] = []
        if req.previous_loan_defaults_on_file == "Yes":
            warnings.append("High Risk Flag: Applicant has a prior loan default on bureau record.")
        if loan_percent_income > 0.40:
            warnings.append(f"Elevated Debt Burden: Requested loan represents {loan_percent_income * 100:.0f}% of gross income (>40% threshold).")
        if req.credit_score < 580:
            warnings.append(f"Subprime Credit Tier: Bureau score {req.credit_score} is below standard prime underwriting floors.")
        if req.person_emp_exp < 1:
            warnings.append("Tenure Flag: Employment tenure is less than 1 year.")

        # 6. Build contract response object
        model_name = self.metadata.get("model_name", "smartloan_ann_classifier")
        model_version = self.metadata.get("model_version", "1.0.0")
        model_type = self.metadata.get("model_type", "Artificial Neural Network (Keras)")
        contract_version = "1.0.0"

        response = PredictionResponse(
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            probability_approved=probability_approved,
            decision_threshold=threshold,
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            contract_version=contract_version,
            loan_intent=req.loan_intent,
            derived_features=derived,
            warnings=warnings
        )

        audit_payload = {
            "loan_intent": req.loan_intent,
            "model_name": model_name,
            "model_version": model_version,
            "contract_version": contract_version,
            "decision_threshold": threshold,
            "predicted_class": predicted_class,
            "probability_approved": probability_approved
        }

        return response, audit_payload


# Global singleton service instance
model_service = ModelService()
