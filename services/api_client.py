import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone


class SmartLoanAPIClient:
    """
    HTTP Client for SmartLoan AI FastAPI Backend adhering to API Contract v1.0.0.
    Supports a seamless 'Provisional Mock Mode' for offline simulation.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0, mock_mode: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.mock_mode = mock_mode
        self._mock_assessments: List[Dict[str, Any]] = [
            {
                "id": 101,
                "loan_intent": "EDUCATION",
                "model_name": "smartloan_ann_classifier (Mock)",
                "model_version": "1.0.0",
                "contract_version": "1.0.0",
                "decision_threshold": 0.50,
                "predicted_class": 1,
                "probability_approved": 0.8842,
                "created_at": "2026-11-21T06:10:00+00:00"
            },
            {
                "id": 102,
                "loan_intent": "PERSONAL",
                "model_name": "smartloan_ann_classifier (Mock)",
                "model_version": "1.0.0",
                "contract_version": "1.0.0",
                "decision_threshold": 0.50,
                "predicted_class": 0,
                "probability_approved": 0.2150,
                "created_at": "2026-11-21T06:30:00+00:00"
            }
        ]

    def health_check(self) -> Dict[str, Any]:
        """Call GET /health or return mock health status."""
        if self.mock_mode:
            return {
                "status": "healthy",
                "contract_version": "1.0.0",
                "model_loaded": True,
                "model_name": "smartloan_ann_classifier (Mock)",
                "model_version": "1.0.0"
            }
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {
                "status": "offline",
                "contract_version": "1.0.0",
                "model_loaded": False,
                "error": str(e)
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Call GET /api/v1/model-info or return mock model benchmarks."""
        if self.mock_mode:
            return {
                "model_name": "smartloan_ann_classifier (Mock)",
                "model_version": "1.0.0",
                "model_type": "Artificial Neural Network (Keras / Mock)",
                "contract_version": "1.0.0",
                "default_threshold": 0.50,
                "supported_loan_categories": [
                    "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"
                ],
                "features_used": {
                    "categorical": ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"],
                    "numeric_raw": ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"],
                    "numeric_derived": ["calc_debt_to_income", "estimated_annual_interest"]
                },
                "evaluation_metrics": {
                    "Best_ANN_Hyperparameters": {"units_1": 128, "units_2": 64, "dropout_rate": 0.25, "lr": 0.0005},
                    "Random_Forest_Baseline": {"accuracy": 0.8969, "precision": 0.7094, "recall": 0.9080, "f1_score": 0.7965, "auc_roc": 0.9733},
                    "Artificial_Neural_Network": {"accuracy": 0.8850, "precision": 0.6834, "recall": 0.8993, "f1_score": 0.7766, "auc_roc": 0.9673}
                }
            }

        resp = requests.get(f"{self.base_url}/api/v1/model-info", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def predict_loan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call POST /api/v1/predict with the 9 contract fields or simulate locally in mock mode.
        """
        if self.mock_mode:
            income = max(float(payload.get("person_income", 50000.0)), 1.0)
            loan_amnt = float(payload.get("loan_amnt", 10000.0))
            rate = float(payload.get("loan_int_rate", 10.0))
            score = int(payload.get("credit_score", 650))
            defaults = str(payload.get("previous_loan_defaults_on_file", "No")).strip().capitalize()
            threshold = float(payload.get("decision_threshold", 0.50))
            intent = str(payload.get("loan_intent", "PERSONAL")).upper()

            loan_percent_income = round(loan_amnt / income, 2)
            calc_debt_to_income = round(loan_amnt / (income + 1.0), 4)
            estimated_annual_interest = round(loan_amnt * (rate / 100.0), 2)

            # Realistic heuristic probability for mock underwriting
            base_score = 0.55
            if score >= 750:
                base_score += 0.30
            elif score >= 670:
                base_score += 0.15
            elif score < 580:
                base_score -= 0.35

            if defaults in {"Yes", "Y"}:
                base_score -= 0.35

            if loan_percent_income > 0.40:
                base_score -= 0.20
            elif loan_percent_income < 0.15:
                base_score += 0.10

            prob = max(0.02, min(0.98, round(base_score, 4)))
            predicted_class = 1 if prob >= threshold else 0
            predicted_label = "APPROVED" if predicted_class == 1 else "REJECTED"

            warnings = []
            if defaults in {"Yes", "Y"}:
                warnings.append("High Risk Flag: Applicant has a prior loan default on bureau record.")
            if loan_percent_income > 0.40:
                warnings.append(f"Elevated Debt Burden: Requested loan represents {loan_percent_income * 100:.0f}% of gross income (>40% threshold).")
            if score < 580:
                warnings.append(f"Subprime Credit Tier: Bureau score {score} is below standard prime underwriting floors.")

            mock_res = {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "probability_approved": prob,
                "decision_threshold": threshold,
                "model_name": "smartloan_ann_classifier (Mock)",
                "model_version": "1.0.0",
                "model_type": "Artificial Neural Network (Keras / Mock)",
                "contract_version": "1.0.0",
                "loan_intent": intent,
                "derived_features": {
                    "loan_percent_income": loan_percent_income,
                    "calc_debt_to_income": calc_debt_to_income,
                    "estimated_annual_interest": estimated_annual_interest
                },
                "warnings": warnings
            }

            # Append to mock audit history
            new_id = len(self._mock_assessments) + 101
            self._mock_assessments.insert(0, {
                "id": new_id,
                "loan_intent": intent,
                "model_name": "smartloan_ann_classifier (Mock)",
                "model_version": "1.0.0",
                "contract_version": "1.0.0",
                "decision_threshold": threshold,
                "predicted_class": predicted_class,
                "probability_approved": prob,
                "created_at": datetime.now(timezone.utc).isoformat()
            })

            return mock_res

        resp = requests.post(f"{self.base_url}/api/v1/predict", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience alias for predict_loan."""
        return self.predict_loan(payload)

    def get_assessments(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Call GET /api/v1/assessments or return mock assessment audit records."""
        if self.mock_mode:
            return self._mock_assessments[offset : offset + limit]

        resp = requests.get(
            f"{self.base_url}/api/v1/assessments",
            params={"limit": limit, "offset": offset},
            timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_assessment_by_id(self, assessment_id: int) -> Optional[Dict[str, Any]]:
        """Call GET /api/v1/assessments/{assessment_id} or return matching mock record."""
        if self.mock_mode:
            for item in self._mock_assessments:
                if item["id"] == assessment_id:
                    return item
            return None

        resp = requests.get(f"{self.base_url}/api/v1/assessments/{assessment_id}", timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
