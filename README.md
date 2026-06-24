# CyberSentinel-AI

End-to-end AI/ML threat detection system integrating four machine learning models
with Microsoft Sentinel for real-time network intrusion detection.

## Architecture

Training pipeline (free — Google Colab):
CIC-IDS2017 → Feature Engineering → 4 ML Models → Ensemble

Production pipeline (Azure credits):
FastAPI Server → Azure Event Hubs → Microsoft Sentinel → Grafana Dashboard

## Models

| Model | Type | Strength |
|---|---|---|
| Isolation Forest | Unsupervised anomaly | Detects zero-day patterns |
| Random Forest | Supervised classifier | High accuracy on known attacks |
| Autoencoder | Deep learning anomaly | Detects subtle deviations |
| LSTM | Sequential pattern | Detects time-series attack chains |

## Dataset

CIC-IDS2017 — Canadian Institute for Cybersecurity
- 2,830,540 network flows
- 80 features per flow
- 14 attack categories (DoS, DDoS, Brute Force, Web Attacks, Botnet, Port Scan)

## Stack

Python 3.10 · Scikit-learn · PyTorch · FastAPI · SHAP ·
Azure Sentinel · Azure Event Hubs · Azure Functions · Grafana · Docker

## Phases

- [x] Phase 1: Foundation and dataset EDA
- [ ] Phase 2: Data engineering and feature pipeline
- [ ] Phase 3: ML model training and SHAP explainability
- [ ] Phase 4: FastAPI inference server and Docker
- [ ] Phase 5: Azure Sentinel integration and KQL rules
- [ ] Phase 6: Grafana dashboard and portfolio documentation