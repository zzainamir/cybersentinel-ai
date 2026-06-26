"""
Predictor class — loads all four trained models and runs ensemble predictions.

Loaded once when the FastAPI server starts. All prediction endpoints
call methods on this single shared instance.
"""

import json
import logging
import numpy as np
import joblib
import torch
from pathlib import Path

from src.models.autoencoder import ThreatAutoencoder
from src.models.lstm        import ThreatLSTM

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s')
logger = logging.getLogger(__name__)


class ThreatPredictor:
    """
    Loads all four models and the ensemble config, then exposes
    predict_single() and predict_batch() for the API endpoints.
    """

    THREAT_LEVELS = [
        (0.25, 'LOW'),
        (0.50, 'MEDIUM'),
        (0.75, 'HIGH'),
        (1.01, 'CRITICAL'),
    ]

    def __init__(self, models_dir: str = 'models/saved',
                 metadata_path: str = 'data/processed/feature_metadata.json'):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {self.device}")

        d = Path(models_dir)

        # ── sklearn models ────────────────────────────────────────────────────
        logger.info("Loading Isolation Forest...")
        self.iso_forest = joblib.load(d / 'isolation_forest.pkl')

        logger.info("Loading Random Forest...")
        self.rf = joblib.load(d / 'random_forest.pkl')

        # ── Autoencoder ───────────────────────────────────────────────────────
        logger.info("Loading Autoencoder...")
        ae_ckpt        = torch.load(d / 'autoencoder.pt', map_location=self.device)
        self.ae        = ThreatAutoencoder(ae_ckpt['input_dim'],
                                           ae_ckpt['latent_dim']).to(self.device)
        self.ae.load_state_dict(ae_ckpt['state_dict'])
        self.ae.eval()
        self.ae_threshold = ae_ckpt['threshold']

        # ── LSTM ──────────────────────────────────────────────────────────────
        logger.info("Loading LSTM...")
        lstm_ckpt  = torch.load(d / 'lstm.pt', map_location=self.device)
        self.lstm  = ThreatLSTM(lstm_ckpt['input_dim'],
                                lstm_ckpt['hidden_size'],
                                lstm_ckpt['num_layers']).to(self.device)
        self.lstm.load_state_dict(lstm_ckpt['state_dict'])
        self.lstm.eval()
        self.seq_len = lstm_ckpt['seq_len']

        # ── Ensemble config ───────────────────────────────────────────────────
        with open(d / 'ensemble_config.json') as f:
            cfg = json.load(f)
        self.weights           = cfg['weights']
        self.ensemble_threshold = cfg['threshold']

        # ── Feature metadata ──────────────────────────────────────────────────
        with open(metadata_path) as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']
        self.n_features    = len(self.feature_names)

        self.is_ready = True
        logger.info(f"All models loaded — {self.n_features} features expected")

    # ── Public prediction methods ─────────────────────────────────────────────

    def predict_single(self, features: list) -> dict:
        """Score one network flow. Returns dict matching ThreatPrediction schema."""
        X = self._validate(features)
        scores = self._score_batch(X)
        return self._format_single(scores, 0)

    def predict_batch(self, flows: list) -> list:
        """Score a list of flows. Returns list of dicts."""
        X = np.array([self._validate(f).squeeze() for f in flows],
                     dtype=np.float32)
        scores = self._score_batch(X)
        return [self._format_single(scores, i) for i in range(len(flows))]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate(self, features: list) -> np.ndarray:
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, received {X.shape[1]}. "
                f"See /models/info for the full feature list."
            )
        return X

    def _minmax(self, arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    def _score_batch(self, X: np.ndarray) -> dict:
        """Run all four models and return normalised score arrays."""
        n = len(X)

        # Isolation Forest
        if_raw  = -self.iso_forest.score_samples(X)
        if_norm = self._minmax(if_raw)

        # Random Forest
        rf_raw  = self.rf.predict_proba(X)[:, 1]
        rf_norm = self._minmax(rf_raw)

        # Autoencoder
        X_t     = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon   = self.ae(X_t)
        ae_raw  = ((X_t - recon) ** 2).mean(dim=1).cpu().numpy()
        ae_norm = self._minmax(ae_raw)

        # LSTM — create a sequence padded with the first row if needed
        if n >= self.seq_len:
            X_seqs = np.array(
                [X[i: i + self.seq_len] for i in range(n - self.seq_len + 1)],
                dtype=np.float32
            )
        else:
            # Pad with repeated first row to reach seq_len
            pad     = np.repeat(X[:1], self.seq_len - n, axis=0)
            X_pad   = np.vstack([pad, X])
            X_seqs  = X_pad[np.newaxis, :]   # (1, seq_len, n_features)

        lstm_probs = np.zeros(n)
        with torch.no_grad():
            seqs_t = torch.tensor(X_seqs, dtype=torch.float32).to(self.device)
            probs  = self.lstm(seqs_t).cpu().numpy()
        lstm_probs[-len(probs):] = probs
        lstm_norm = self._minmax(lstm_probs)

        # Weighted ensemble
        ensemble = (
            self.weights['iso']  * if_norm  +
            self.weights['rf']   * rf_norm  +
            self.weights['ae']   * ae_norm  +
            self.weights['lstm'] * lstm_norm
        )
        return {
            'ensemble': ensemble,
            'if':       if_norm,
            'rf':       rf_norm,
            'ae':       ae_norm,
            'lstm':     lstm_norm,
        }

    def _format_single(self, scores: dict, idx: int) -> dict:
        score = float(scores['ensemble'][idx])
        return {
            'threat_score': round(score, 4),
            'is_attack':    score > self.ensemble_threshold,
            'threat_level': self._threat_level(score),
            'model_scores': {
                'isolation_forest': round(float(scores['if'][idx]),   4),
                'random_forest':    round(float(scores['rf'][idx]),   4),
                'autoencoder':      round(float(scores['ae'][idx]),   4),
                'lstm':             round(float(scores['lstm'][idx]), 4),
            }
        }

    def _threat_level(self, score: float) -> str:
        for threshold, level in self.THREAT_LEVELS:
            if score < threshold:
                return level
        return 'CRITICAL'