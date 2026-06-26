"""
CyberSentinel-AI — FastAPI inference server.

Endpoints:
  GET  /health          — server and model health check
  GET  /models/info     — feature list and model descriptions
  POST /predict         — score a single network flow
  POST /predict/batch   — score up to 1,000 flows at once

Interactive docs available at http://localhost:8000/docs after startup.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    FlowFeatures, BatchFlowFeatures,
    ThreatPrediction, BatchPrediction,
    HealthResponse, ModelInfoResponse,
)
from src.api.predictor import ThreatPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Global predictor — loaded once on startup, shared across all requests
predictor: ThreatPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models when the server starts, log shutdown when it stops."""
    global predictor
    logger.info("=" * 50)
    logger.info("  CyberSentinel-AI starting up")
    logger.info("=" * 50)
    start = time.time()
    predictor = ThreatPredictor()
    logger.info(f"Ready in {time.time() - start:.1f} s")
    yield
    logger.info("CyberSentinel-AI shutting down")


app = FastAPI(
    title="CyberSentinel-AI",
    description=(
        "Multi-model AI/ML threat detection system combining Isolation Forest, "
        "Random Forest, Autoencoder, and LSTM into a weighted ensemble. "
        "Trained on the CIC-IDS2017 dataset (2.8M network flows, 14 attack types)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: log every request with timing ─────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path}  "
                f"{response.status_code}  {elapsed:.1f}ms")
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Check whether the server is running and all models are loaded.
    Use this as your readiness probe in Azure Container Instances (Phase 5).
    """
    ready = predictor is not None and predictor.is_ready
    return {
        "status":        "healthy" if ready else "degraded",
        "models_loaded": ready,
        "n_features":    predictor.n_features if ready else 0,
        "model_list":    ["isolation_forest", "random_forest",
                          "autoencoder", "lstm"],
    }


@app.get("/models/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """
    Return the full feature list and descriptions of each model.
    Use this to understand what feature vector your /predict calls should send.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return {
        "version":       "1.0.0",
        "n_features":    predictor.n_features,
        "feature_names": predictor.feature_names,
        "models": {
            "isolation_forest": "Unsupervised anomaly — trained on benign-only data",
            "random_forest":    "Supervised classifier — 200 trees, class_weight=balanced",
            "autoencoder":      "Deep learning anomaly — reconstruction error threshold",
            "lstm":             "Sequential pattern — sliding window over flow sequences",
        },
        "ensemble": (
            "Weighted sum of normalised scores. "
            "Weights are proportional to validation F1 of each model."
        ),
    }


@app.post("/predict", response_model=ThreatPrediction, tags=["Detection"])
def predict_single(flow: FlowFeatures):
    """
    Score a single network flow.

    Send a feature vector of exactly n_features floats (see /models/info).
    Returns a threat score (0–1), binary classification, threat level,
    and the individual score from each of the four models.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    try:
        return predictor.predict_single(flow.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Detection"])
def predict_batch(batch: BatchFlowFeatures):
    """
    Score a batch of up to 1,000 network flows in a single request.

    More efficient than calling /predict repeatedly.
    Returns a summary (flow count, attack count, attack rate)
    and the full prediction for every flow.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    try:
        predictions  = predictor.predict_batch(batch.flows)
        attack_count = sum(1 for p in predictions if p['is_attack'])
        return {
            "flow_count":   len(predictions),
            "attack_count": attack_count,
            "attack_rate":  round(attack_count / len(predictions) * 100, 2),
            "predictions":  predictions,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))