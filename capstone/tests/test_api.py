"""
test_api.py
Part D: Functional testing of the end-to-end system (data -> model -> API)
Run with: pytest tests/test_api.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    """The API must respond with status ok and the model must load successfully"""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_positive_text():
    """Clearly positive text should be classified as positive"""
    resp = client.post("/api/predict", json={"text": "I love this, it's amazing and works perfectly!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"positive", "negative", "neutral"}


def test_predict_negative_text_thai():
    """Test negative Thai-language text"""
    resp = client.post("/api/predict", json={"text": "Terrible, the product broke after just one day of use"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "negative"


def test_predict_empty_text_rejected():
    """Empty text must be rejected with a validation error (422)"""
    resp = client.post("/api/predict", json={"text": "   "})
    assert resp.status_code == 422


def test_predict_missing_field():
    """Not sending the 'text' field at all must return 422"""
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 422


def test_batch_predict():
    """Test the batch prediction endpoint with multiple texts at once"""
    resp = client.post("/api/predict/batch", json={"texts": ["Great service!", "Terrible product.", "It's okay."]})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 3


def test_batch_predict_filters_blank_entries():
    """Blank entries in the list should be skipped, without failing the whole request"""
    resp = client.post("/api/predict/batch", json={"texts": ["Good product", "   "]})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_metrics_endpoint():
    """Check that the metrics endpoint returns final_test_metrics that meet the threshold set in Part A"""
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    metrics = resp.json()["final_test_metrics"]
    assert metrics["accuracy"] >= 0.70  # success criterion from Part A
