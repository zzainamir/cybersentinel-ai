"""
CyberSentinel-AI — FastAPI inference server.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    FlowFeatures, BatchFlowFeatures,
    ThreatPrediction, BatchPrediction,
    HealthResponse, ModelInfoResponse,
)
from src.api.predictor import ThreatPredictor
from src.api.eventhub_sender import EventHubSender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Global instances
predictor: ThreatPredictor | None = None
sender: EventHubSender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize sender on startup."""
    global predictor, sender

    logger.info("=" * 50)
    logger.info("  CyberSentinel-AI starting up")
    logger.info("=" * 50)

    # Debug — confirm connection string is loading from .env
    cs = os.environ.get('EVENTHUB_CONNECTION_STRING', 'NOT FOUND')
    logger.info(f"Event Hubs connection string: {cs[:60]}")

    start = time.time()

    predictor = ThreatPredictor()
    sender    = EventHubSender()

    logger.info(f"Ready in {time.time() - start:.1f} s")
    yield

    logger.info("CyberSentinel-AI shutting down")


app = FastAPI(
    title="CyberSentinel-AI",
    description=(
        "Multi-model AI/ML threat detection system combining Isolation Forest, "
        "Random Forest, Autoencoder, and LSTM into a weighted ensemble."
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
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return {
        "version":       "1.0.0",
        "n_features":    predictor.n_features,
        "feature_names": predictor.feature_names,
        "models": {
            "isolation_forest": "Unsupervised anomaly",
            "random_forest":    "Supervised classifier",
            "autoencoder":      "Reconstruction error",
            "lstm":             "Sequential pattern",
        },
        "ensemble": "Weighted sum of model scores",
    }


@app.post("/predict", response_model=ThreatPrediction, tags=["Detection"])
def predict_single(flow: FlowFeatures):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    try:
        result = predictor.predict_single(flow.features)
        if sender and result['threat_level'] in ('HIGH', 'CRITICAL'):
            sender.send_if_threat(result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchPrediction, tags=["Detection"])
def predict_batch(batch: BatchFlowFeatures):
    if not predictor:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    try:
        predictions = predictor.predict_batch(batch.flows)
        if sender:
            for result in predictions:
                if result['threat_level'] in ('HIGH', 'CRITICAL'):
                    sender.send_if_threat(result)
        attack_count = sum(1 for p in predictions if p['is_attack'])
        return {
            "flow_count":   len(predictions),
            "attack_count": attack_count,
            "attack_rate":  round(attack_count / len(predictions) * 100, 2),
            "predictions":  predictions,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))