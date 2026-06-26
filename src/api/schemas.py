"""
Request and response schemas for the CyberSentinel-AI API.

FastAPI uses these Pydantic models to:
  1. Validate incoming request data automatically
  2. Generate the interactive /docs documentation page
  3. Serialise response data to JSON
"""

from pydantic import BaseModel, Field
from typing import Dict, List


class FlowFeatures(BaseModel):
    """A single network flow submitted for threat analysis."""
    features: List[float] = Field(
        ...,
        description="Network flow feature vector. Must match the number of "
                    "features used during training (see /models/info).",
        min_length=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [0.0, 1.2, -0.5, 3.1]  # placeholder values
            }
        }
    }


class BatchFlowFeatures(BaseModel):
    """A batch of network flows submitted together for threat analysis."""
    flows: List[List[float]] = Field(
        ...,
        description="List of feature vectors — each inner list is one flow.",
        min_length=1,
        max_length=1000,
    )


class ModelScores(BaseModel):
    """Individual threat score from each model (0 = safe, 1 = threat)."""
    isolation_forest: float
    random_forest:    float
    autoencoder:      float
    lstm:             float


class ThreatPrediction(BaseModel):
    """Threat analysis result for a single network flow."""
    threat_score:  float       # 0.0 (safe) to 1.0 (definite threat)
    is_attack:     bool        # True if threat_score exceeds ensemble threshold
    threat_level:  str         # LOW / MEDIUM / HIGH / CRITICAL
    model_scores:  ModelScores # individual scores from each of the four models


class BatchPrediction(BaseModel):
    """Threat analysis results for a batch of flows."""
    flow_count:   int
    attack_count: int
    attack_rate:  float                  # percentage of flows flagged
    predictions:  List[ThreatPrediction]


class HealthResponse(BaseModel):
    """Server health status."""
    status:        str   # "healthy" or "degraded"
    models_loaded: bool
    n_features:    int
    model_list:    List[str]


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded models."""
    version:       str
    n_features:    int
    feature_names: List[str]
    models:        Dict[str, str]
    ensemble:      str