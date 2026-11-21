import os
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    AssessmentAuditRecord
)
from backend.database import (
    init_db,
    insert_assessment,
    get_assessments,
    get_assessment_by_id
)
from backend.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize SQLite DB and load model artifacts
    init_db()
    try:
        model_service.load_artifacts()
    except Exception as e:
        print(f"[WARNING] Model loading deferred: {e}")
    yield
    # Shutdown logic if needed


app = FastAPI(
    title="SmartLoan AI Credit Underwriting API",
    description="FastAPI Backend for SmartLoan AI implementing API Contract v1.0.0",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit frontend and local integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Service health check endpoint.
    Returns status, contract version, and model readiness.
    """
    if not model_service.is_loaded:
        try:
            model_service.load_artifacts()
        except Exception as e:
            print(f"[WARNING] Health check artifact load failed: {e}")

    return HealthResponse(
        status="healthy",
        contract_version="1.0.0",
        model_loaded=model_service.is_loaded,
        model_name=model_service.metadata.get("model_name", "smartloan_ann_classifier"),
        model_version=model_service.metadata.get("model_version", "1.0.0")
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """
    Returns model metadata, supported categories, feature definitions,
    and unseen test set evaluation benchmarks.
    """
    if not model_service.is_loaded:
        model_service.load_artifacts()

    supported_intents = model_service.schema.get("cat_values", {}).get(
        "loan_intent",
        ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    )

    features_used = model_service.metadata.get("features_used", {
        "categorical": model_service.schema.get("cat_cols", []),
        "numeric_raw": model_service.schema.get("raw_num_cols", []),
        "numeric_derived": ["calc_debt_to_income", "estimated_annual_interest"]
    })

    return ModelInfoResponse(
        model_name=model_service.metadata.get("model_name", "smartloan_ann_classifier"),
        model_version=model_service.metadata.get("model_version", "1.0.0"),
        model_type=model_service.metadata.get("model_type", "Artificial Neural Network (Keras)"),
        contract_version="1.0.0",
        default_threshold=float(model_service.metadata.get("default_threshold", 0.50)),
        supported_loan_categories=supported_intents,
        features_used=features_used,
        evaluation_metrics=model_service.metrics
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_loan(request: PredictionRequest):
    """
    Underwrite a loan application adhering to Contract v1.0.0.
    Validates 9 frontend input fields, computes derived ratios,
    runs neural network inference, and logs non-sensitive audit record to SQLite.
    """
    try:
        response, audit_payload = model_service.predict(request)

        # Store privacy-safe non-sensitive assessment audit in SQLite (smartloan.db)
        insert_assessment(
            loan_intent=audit_payload["loan_intent"],
            model_name=audit_payload["model_name"],
            model_version=audit_payload["model_version"],
            contract_version=audit_payload["contract_version"],
            decision_threshold=audit_payload["decision_threshold"],
            predicted_class=audit_payload["predicted_class"],
            probability_approved=audit_payload["probability_approved"]
        )

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )


@app.get("/api/v1/assessments", response_model=List[AssessmentAuditRecord], tags=["Audit"])
def list_assessments(
    limit: int = Query(100, ge=1, le=500, description="Max audit records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Retrieve historical underwriting audit records from SQLite (smartloan.db).
    Strictly privacy-safe: Contains NO applicant PII, income, age, or credit scores.
    """
    records = get_assessments(limit=limit, offset=offset)
    return [AssessmentAuditRecord(**r) for r in records]


@app.get("/api/v1/assessments/{assessment_id}", response_model=AssessmentAuditRecord, tags=["Audit"])
def get_assessment(assessment_id: int):
    """
    Retrieve a specific non-sensitive assessment audit record by its unique ID.
    Returns 404 if the record does not exist.
    """
    record = get_assessment_by_id(assessment_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment audit record with ID {assessment_id} not found."
        )
    return AssessmentAuditRecord(**record)
