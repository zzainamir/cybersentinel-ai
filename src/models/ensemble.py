"""
Ensemble combining Isolation Forest, Random Forest,
Autoencoder, and LSTM into a single weighted threat score.
"""
import json
import numpy as np
import joblib
import torch

from src.models.autoencoder import ThreatAutoencoder
from src.models.lstm        import ThreatLSTM


class ThreatEnsemble:
    """
    Loads all four trained models and produces a combined threat score.

    Usage:
        ensemble = ThreatEnsemble('models/saved')
        score, label = ensemble.predict(feature_vector)
    """

    def __init__(self, models_dir: str = 'models/saved'):
        from pathlib import Path
        d = Path(models_dir)

        # Load sklearn models
        self.iso_forest = joblib.load(d / 'isolation_forest.pkl')
        self.rf         = joblib.load(d / 'random_forest.pkl')

        # Load autoencoder
        ae_ckpt   = torch.load(d / 'autoencoder.pt', map_location='cpu')
        self.ae   = ThreatAutoencoder(ae_ckpt['input_dim'], ae_ckpt['latent_dim'])
        self.ae.load_state_dict(ae_ckpt['state_dict'])
        self.ae.eval()
        self.ae_threshold = ae_ckpt['threshold']

        # Load LSTM
        lstm_ckpt  = torch.load(d / 'lstm.pt', map_location='cpu')
        self.lstm  = ThreatLSTM(lstm_ckpt['input_dim'],
                                lstm_ckpt['hidden_size'],
                                lstm_ckpt['num_layers'])
        self.lstm.load_state_dict(lstm_ckpt['state_dict'])
        self.lstm.eval()
        self.seq_len = lstm_ckpt['seq_len']

        # Load ensemble config
        with open(d / 'ensemble_config.json') as f:
            cfg = json.load(f)
        self.weights   = cfg['weights']
        self.threshold = cfg['threshold']

    def _minmax(self, arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    def predict_batch(self, X: np.ndarray) -> dict:
        """
        Score a batch of feature vectors.
        Returns dict with individual scores and ensemble threat score.
        """
        # IF score (higher = more anomalous)
        if_score  = self._minmax(-self.iso_forest.score_samples(X))

        # RF probability of attack
        rf_score  = self._minmax(self.rf.predict_proba(X)[:, 1])

        # Autoencoder reconstruction error
        X_t       = torch.tensor(X, dtype=torch.float32)
        ae_score  = self._minmax(self.ae.reconstruction_error(X_t).numpy())

        # LSTM — create sequences from consecutive rows
        X_seqs    = np.array([X[i: i + self.seq_len]
                               for i in range(len(X) - self.seq_len + 1)],
                              dtype=np.float32)
        lstm_probs = np.zeros(len(X))
        if len(X_seqs) > 0:
            with torch.no_grad():
                lstm_p = self.lstm(torch.tensor(X_seqs)).numpy()
            lstm_probs[-len(lstm_p):] = lstm_p
        lstm_score = self._minmax(lstm_probs)

        # Weighted ensemble
        ensemble = (
            self.weights['iso']  * if_score  +
            self.weights['rf']   * rf_score  +
            self.weights['ae']   * ae_score  +
            self.weights['lstm'] * lstm_score
        )
        return {
            'threat_score':      ensemble,
            'is_attack':         (ensemble > self.threshold).astype(int),
            'score_if':          if_score,
            'score_rf':          rf_score,
            'score_ae':          ae_score,
            'score_lstm':        lstm_score,
        }