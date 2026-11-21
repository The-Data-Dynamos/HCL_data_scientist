from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Contract v1.0.0 Request Schema.
    Validates all 9 required frontend fields.
    """
    person_income: float = Field(..., gt=0, description="Applicant gross annual income in USD (> 0)")
    person_emp_exp: float = Field(..., ge=0, le=80, description="Employment experience in years (>= 0)")
    person_home_ownership: str = Field(..., description="Home ownership status: RENT, OWN, MORTGAGE, OTHER")
    loan_amnt: float = Field(..., gt=0, description="Requested loan principal in USD (> 0)")
    loan_intent: str = Field(..., description="Loan purpose/intent category")
    loan_int_rate: float = Field(..., gt=0, le=100, description="Interest rate percentage (> 0 and <= 100)")
    cb_person_cred_hist_length: float = Field(..., ge=0, le=80, description="Credit bureau history in years (>= 0)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit bureau score (300 to 850)")
    previous_loan_defaults_on_file: str = Field(..., description="Prior loan defaults on file: Yes, No, Y, N")
    decision_threshold: Optional[float] = Field(0.50, ge=0.0, le=1.0, description="Underwriting decision threshold")

    @field_validator("person_home_ownership")
    @classmethod
    def validate_home_ownership(cls, v: str) -> str:
        clean = v.strip().upper()
        allowed = {"RENT", "OWN", "MORTGAGE", "OTHER"}
        if clean not in allowed:
            raise ValueError(f"person_home_ownership must be one of {sorted(allowed)}, got: '{v}'")
        return clean

    @field_validator("loan_intent")
    @classmethod
    def validate_loan_intent(cls, v: str) -> str:
        clean = v.strip().upper()
        allowed = {"DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"}
        if clean not in allowed:
            raise ValueError(f"loan_intent must be one of {sorted(allowed)}, got: '{v}'")
        return clean

    @field_validator("previous_loan_defaults_on_file")
    @classmethod
    def validate_defaults(cls, v: str) -> str:
        clean = v.strip().capitalize()
        if clean in {"Y", "Yes"}:
            return "Yes"
        if clean in {"N", "No"}:
            return "No"
        raise ValueError(f"previous_loan_defaults_on_file must be 'Yes', 'No', 'Y', or 'N', got: '{v}'")


class DerivedFeatures(BaseModel):
    """Derived financial metrics calculated during prediction."""
    loan_percent_income: float = Field(..., description="Loan principal divided by gross income, rounded to 2 decimal places")
    calc_debt_to_income: float = Field(..., description="Debt-to-income ratio (loan / (income + 1))")
    estimated_annual_interest: float = Field(..., description="Estimated annual interest charge in USD")


class PredictionResponse(BaseModel):
    """
    Contract v1.0.0 Prediction Response Schema.
    Strictly aligns with the 11 contract response keys.
    """
    predicted_class: int = Field(..., description="Binary decision: 1 = Approved, 0 = Rejected")
    predicted_label: str = Field(..., description="Human-readable decision: APPROVED or REJECTED")
    probability_approved: float = Field(..., ge=0.0, le=1.0, description="Model approval probability")
    decision_threshold: float = Field(..., ge=0.0, le=1.0, description="Applied decision cutoff threshold")
    model_name: str = Field(..., description="Registered model identifier")
    model_version: str = Field(..., description="Model artifact version")
    model_type: str = Field(..., description="Algorithm family / architecture")
    contract_version: str = Field("1.0.0", description="API Contract specification version")
    loan_intent: str = Field(..., description="Applicant loan intent")
    derived_features: DerivedFeatures = Field(..., description="Engineered features computed from inputs")
    warnings: List[str] = Field(default_factory=list, description="Automated policy and risk warning flags")


class HealthResponse(BaseModel):
    """GET /health response schema."""
    status: str = Field("healthy", description="Service health state")
    contract_version: str = Field("1.0.0", description="API Contract specification version")
    model_loaded: bool = Field(..., description="Whether inference models are initialized and ready")
    model_name: str = Field(..., description="Primary inference model name")
    model_version: str = Field(..., description="Primary inference model version")


class AssessmentAuditRecord(BaseModel):
    """
    Non-sensitive assessment audit record stored in SQLite (smartloan.db).
    Strictly privacy-safe: NO applicant income, age, gender, or credit score stored.
    """
    id: int = Field(..., description="Audit record unique identifier")
    loan_intent: str = Field(..., description="Loan intent category")
    model_name: str = Field(..., description="Model name used for underwriting")
    model_version: str = Field(..., description="Model version")
    contract_version: str = Field("1.0.0", description="Contract version")
    decision_threshold: float = Field(..., description="Threshold applied")
    predicted_class: int = Field(..., description="Predicted class (1 or 0)")
    probability_approved: float = Field(..., description="Approval probability")
    created_at: str = Field(..., description="UTC ISO-8601 timestamp")


class ModelInfoResponse(BaseModel):
    """GET /api/v1/model-info response schema."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type / framework")
    contract_version: str = Field("1.0.0", description="API Contract version")
    default_threshold: float = Field(0.50, description="Default policy decision cutoff")
    supported_loan_categories: List[str] = Field(..., description="Supported loan intent categories")
    features_used: Dict[str, Any] = Field(..., description="Feature breakdown: categorical, numeric_raw, numeric_derived")
    evaluation_metrics: Dict[str, Any] = Field(..., description="Test set performance metrics for ANN and RF baseline")
