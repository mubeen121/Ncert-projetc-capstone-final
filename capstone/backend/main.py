"""
main.py — Sentiment Analysis Dashboard Backend API (FastAPI)
==============================================================
Part C: API/Backend Development

Endpoints:
  GET  /api/health           -> Check server and model status
  POST /api/predict          -> Analyze the sentiment of a single text
  POST /api/predict/batch    -> Analyze the sentiment of multiple texts at once
  GET  /api/metrics          -> Return the model's latest metrics (from model/metrics.json)

Note on Authentication:
This project is a public sentiment-analysis utility that does not store any
personal data or user accounts, so a login/JWT system is "not required"
(out of scope as stated in Part A) — if per-user search history needs to be
stored in the future, a JWT + HTTP-only cookie authentication system would be added.
"""
import json
import os
import time
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

app = FastAPI(
    title="Sentiment Analysis Dashboard API",
    description="Capstone Project (PKCERT Task 28) - Analyze the sentiment of review text",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None


def get_model_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model file not found. Please run model/train_model.py first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="The text to analyze the sentiment of")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def run_prediction(text: str) -> PredictResponse:
    bundle = get_model_bundle()
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    start = time.perf_counter()
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    labels = model.classes_
    label_probs = {label: round(float(p), 4) for label, p in zip(labels, proba)}
    best_idx = proba.argmax()
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    return PredictResponse(
        text=text,
        label=str(labels[best_idx]),
        confidence=round(float(proba[best_idx]), 4),
        probabilities=label_probs,
        latency_ms=latency_ms,
    )


@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        get_model_bundle()
        loaded = True
    except HTTPException:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        return run_prediction(payload.text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/api/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest):
    results = []
    for text in payload.texts:
        clean = text.strip()
        if not clean:
            continue
        results.append(run_prediction(clean))
    if not results:
        raise HTTPException(status_code=400, detail="No valid, non-empty texts were provided.")
    return BatchPredictResponse(results=results)


@app.get("/api/metrics")
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found. Please run model/train_model.py first.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def root():
    return {"message": "Sentiment Analysis Dashboard API is running. See /docs for API documentation."}
