import pytest
from fastapi.testclient import TestClient
from backend.main import app
from services.api_client import SmartLoanAPIClient


@pytest.fixture(scope="module")
def client():
    """Test client for FastAPI backend."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Valid 9-field input payload strictly adhering to Contract v1.0.0."""
    return {
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


def test_1_health_check_contract(client):
    """1. GET /health adheres strictly to API Contract v1.0.0."""
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert data["status"] == "healthy"
    assert data["contract_version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert "model_name" in data
    assert "model_version" in data


def test_2_model_info_contract(client):
    """2. GET /api/v1/model-info returns complete metadata & metrics without fabricating."""
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert data["contract_version"] == "1.0.0"
    assert data["model_name"] == "smartloan_ann_classifier"
    assert data["model_version"] == "1.0.0"
    assert "Artificial Neural Network" in data["model_type"]
    assert data["default_threshold"] == 0.50

    expected_intents = {"DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"}
    assert set(data["supported_loan_categories"]) == expected_intents

    assert "evaluation_metrics" in data
    assert "Artificial_Neural_Network" in data["evaluation_metrics"]
    assert "Random_Forest_Baseline" in data["evaluation_metrics"]


def test_3_predict_valid_payload_contract(client, valid_payload):
    """3. POST /api/v1/predict returns exact 11 contract response keys with valid ranges."""
    resp = client.post("/api/v1/predict", json=valid_payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # Exact Contract v1.0.0 Keys
    expected_keys = {
        "predicted_class",
        "predicted_label",
        "probability_approved",
        "decision_threshold",
        "model_name",
        "model_version",
        "model_type",
        "contract_version",
        "loan_intent",
        "derived_features",
        "warnings"
    }
    assert set(data.keys()) == expected_keys

    assert data["predicted_class"] in {0, 1}
    assert data["predicted_label"] in {"APPROVED", "REJECTED"}
    assert 0.0 <= data["probability_approved"] <= 1.0
    assert data["decision_threshold"] == 0.50
    assert data["contract_version"] == "1.0.0"
    assert data["loan_intent"] == "EDUCATION"
    assert isinstance(data["warnings"], list)


def test_4_predict_computes_derived_features(client, valid_payload):
    """4. POST /api/v1/predict computes loan_percent_income = round(loan_amnt / person_income, 2)."""
    # 10000 / 65000 = 0.1538... -> rounded to 0.15
    resp = client.post("/api/v1/predict", json=valid_payload)
    assert resp.status_code == 200
    derived = resp.json()["derived_features"]

    expected_ratio = round(valid_payload["loan_amnt"] / valid_payload["person_income"], 2)
    assert derived["loan_percent_income"] == expected_ratio
    assert derived["loan_percent_income"] == 0.15
    assert "calc_debt_to_income" in derived
    assert "estimated_annual_interest" in derived


def test_5_predict_missing_required_fields(client, valid_payload):
    """5. POST /api/v1/predict returns HTTP 422 if any of the 9 required fields are missing."""
    for field in [
        "person_income", "person_emp_exp", "person_home_ownership",
        "loan_amnt", "loan_intent", "loan_int_rate",
        "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"
    ]:
        incomplete = valid_payload.copy()
        del incomplete[field]
        resp = client.post("/api/v1/predict", json=incomplete)
        assert resp.status_code == 422, f"Expected 422 when missing '{field}', got {resp.status_code}"


def test_6_predict_invalid_data_types_or_ranges(client, valid_payload):
    """6. POST /api/v1/predict rejects invalid values and ranges with 422."""
    # Negative income
    bad_payload = valid_payload.copy()
    bad_payload["person_income"] = -500.0
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422

    # Credit score below 300
    bad_payload = valid_payload.copy()
    bad_payload["credit_score"] = 250
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422

    # Credit score above 850
    bad_payload = valid_payload.copy()
    bad_payload["credit_score"] = 900
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422

    # Invalid loan intent
    bad_payload = valid_payload.copy()
    bad_payload["loan_intent"] = "GAMBLING"
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422

    # Invalid home ownership
    bad_payload = valid_payload.copy()
    bad_payload["person_home_ownership"] = "COOP"
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422


def test_7_assessments_audit_persistence_and_privacy(client, valid_payload):
    """7. GET /api/v1/assessments persists audit trail and strictly protects applicant privacy."""
    # Trigger a prediction to ensure audit record exists
    predict_resp = client.post("/api/v1/predict", json=valid_payload)
    assert predict_resp.status_code == 200

    resp = client.get("/api/v1/assessments")
    assert resp.status_code == 200
    records = resp.json()
    assert isinstance(records, list)
    assert len(records) > 0

    latest = records[0]
    expected_audit_keys = {
        "id", "loan_intent", "model_name", "model_version",
        "contract_version", "decision_threshold", "predicted_class",
        "probability_approved", "created_at"
    }
    assert set(latest.keys()) == expected_audit_keys

    # Privacy by Design: Strictly verify that no applicant PII/income/score is stored
    forbidden_keys = {"person_income", "credit_score", "person_age", "person_gender", "person_emp_exp"}
    for f in forbidden_keys:
        assert f not in latest, f"Privacy violation: sensitive field '{f}' leaked in audit log!"


def test_8_assessment_by_id_and_not_found(client, valid_payload):
    """8. GET /api/v1/assessments/{id} returns matching record, and 404 for unknown ID."""
    predict_resp = client.post("/api/v1/predict", json=valid_payload)
    assert predict_resp.status_code == 200

    list_resp = client.get("/api/v1/assessments")
    assert list_resp.status_code == 200
    latest_id = list_resp.json()[0]["id"]

    # Valid ID lookup
    lookup_resp = client.get(f"/api/v1/assessments/{latest_id}")
    assert lookup_resp.status_code == 200
    data = lookup_resp.json()
    assert data["id"] == latest_id
    assert data["loan_intent"] == valid_payload["loan_intent"]

    # Non-existent ID lookup
    not_found_resp = client.get("/api/v1/assessments/99999999")
    assert not_found_resp.status_code == 404
    assert "not found" in not_found_resp.json()["detail"].lower()


def test_services_api_client_mock_mode(valid_payload):
    """Bonus test: Verify services/api_client.py seamlessly functions in Provisional Mock Mode."""
    client = SmartLoanAPIClient(mock_mode=True)

    health = client.health_check()
    assert health["status"] == "healthy"
    assert health["contract_version"] == "1.0.0"

    info = client.get_model_info()
    assert info["contract_version"] == "1.0.0"

    pred = client.predict_loan(valid_payload)
    assert pred["contract_version"] == "1.0.0"
    assert pred["predicted_class"] in {0, 1}
    assert pred["derived_features"]["loan_percent_income"] == round(valid_payload["loan_amnt"] / valid_payload["person_income"], 2)

    assessments = client.get_assessments()
    assert len(assessments) >= 1
    first_id = assessments[0]["id"]

    lookup = client.get_assessment_by_id(first_id)
    assert lookup is not None
    assert lookup["id"] == first_id

    missing = client.get_assessment_by_id(999999)
    assert missing is None
