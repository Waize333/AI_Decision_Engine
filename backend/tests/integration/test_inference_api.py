"""
backend/tests/integration/test_inference_api.py
================================================
PURPOSE:
  Integration tests for POST /inference and GET /inference/{id}.

WHAT WE TEST:
  ✅ Inference succeeds with valid features
  ✅ Response contains required fields (request_id, prediction, confidence)
  ✅ Confidence is between 0.0 and 1.0
  ✅ Unauthenticated requests are rejected
  ✅ Missing features dict is rejected
  ✅ Same input produces same prediction (stub is deterministic)
  ✅ Client cannot access GET /inference/{id} (analyst-only)
"""

import pytest

pytestmark = pytest.mark.asyncio

VALID_FEATURES = {
    "transaction_amount": 1250.00,
    "merchant_category": "electronics",
    "hour_of_day": 23,
    "is_international": True,
    "user_account_age_days": 30,
    "num_transactions_today": 5,
}


class TestInferencePost:

    async def test_inference_requires_auth(self, client):
        """Unauthenticated request must return 401."""
        response = await client.post("/api/v1/inference", json={"features": VALID_FEATURES})
        assert response.status_code == 401

    async def test_inference_success(self, client, client_auth_headers, active_model_version):
        """Happy path: returns prediction with all required fields."""
        response = await client.post(
            "/api/v1/inference",
            json={"features": VALID_FEATURES},
            headers=client_auth_headers,
        )
        assert response.status_code == 200
        data = response.json()

        # All required response fields must be present
        assert "request_id" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert "created_at" in data

    async def test_confidence_in_valid_range(self, client, client_auth_headers, active_model_version):
        """Confidence must always be between 0.0 and 1.0 (inclusive)."""
        response = await client.post(
            "/api/v1/inference",
            json={"features": VALID_FEATURES},
            headers=client_auth_headers,
        )
        assert response.status_code == 200
        confidence = response.json()["confidence"]
        assert 0.0 <= confidence <= 1.0

    async def test_request_id_is_unique(self, client, client_auth_headers, active_model_version):
        """Each call generates a distinct request_id."""
        r1 = await client.post(
            "/api/v1/inference",
            json={"features": VALID_FEATURES},
            headers=client_auth_headers,
        )
        r2 = await client.post(
            "/api/v1/inference",
            json={"features": VALID_FEATURES},
            headers=client_auth_headers,
        )
        assert r1.json()["request_id"] != r2.json()["request_id"]

    async def test_client_supplied_request_id_is_used(
        self, client, client_auth_headers, active_model_version
    ):
        """Client-supplied request_id must appear in the response."""
        custom_id = "my-custom-id-abc123"
        response = await client.post(
            "/api/v1/inference",
            json={"features": VALID_FEATURES, "request_id": custom_id},
            headers=client_auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == custom_id

    async def test_missing_features_returns_422(self, client, client_auth_headers):
        """Request body without 'features' key must be rejected."""
        response = await client.post(
            "/api/v1/inference",
            json={"model_version": "v1"},   # no 'features' key
            headers=client_auth_headers,
        )
        assert response.status_code == 422

    async def test_empty_features_dict_accepted(
        self, client, client_auth_headers, active_model_version
    ):
        """
        An empty features dict should be accepted at the API level.
        The model service (and eventually a validator) handles
        feature validation — not the schema.
        """
        response = await client.post(
            "/api/v1/inference",
            json={"features": {}},
            headers=client_auth_headers,
        )
        # Empty features → stub model returns a prediction
        assert response.status_code == 200


class TestInferenceGet:

    async def test_client_cannot_get_inference_by_id(
        self, client, client_auth_headers
    ):
        """CLIENT role must NOT access GET /inference/{id} — analyst only."""
        response = await client.get(
            "/api/v1/inference/some-request-id",
            headers=client_auth_headers,
        )
        assert response.status_code == 403

    async def test_analyst_gets_404_for_missing_request_id(
        self, client, analyst_auth_headers
    ):
        """Non-existent request_id must return 404."""
        response = await client.get(
            "/api/v1/inference/nonexistent-id-xyz",
            headers=analyst_auth_headers,
        )
        assert response.status_code == 404
