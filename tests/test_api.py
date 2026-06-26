"""
Automated tests for all CyberSentinel-AI API endpoints.

Run with the server already started:
    python tests/test_api.py

All tests print PASS or FAIL with details.
"""

import requests
import json
import sys

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  —  {detail}")
        FAIL += 1


def get_n_features() -> int:
    r = requests.get(f"{BASE}/health")
    return r.json().get("n_features", 67)


def run_tests():
    print("\nCyberSentinel-AI API tests")
    print("=" * 45)
    n = get_n_features()
    zeros  = [0.0] * n
    # Use large values to simulate potentially anomalous traffic
    highs  = [999.0] * n

    # ── Health ────────────────────────────────────────────────────────────────
    print("\nGET /health")
    r = requests.get(f"{BASE}/health")
    check("Status 200",        r.status_code == 200)
    check("models_loaded true", r.json().get("models_loaded") is True)
    check("n_features > 0",    r.json().get("n_features", 0) > 0)

    # ── Model info ────────────────────────────────────────────────────────────
    print("\nGET /models/info")
    r = requests.get(f"{BASE}/models/info")
    check("Status 200",             r.status_code == 200)
    check("feature_names present",  "feature_names" in r.json())
    check("4 models listed",        len(r.json().get("models", {})) == 4)

    # ── Single prediction ─────────────────────────────────────────────────────
    print("\nPOST /predict  (zero vector — expect LOW threat)")
    r = requests.post(f"{BASE}/predict", json={"features": zeros})
    check("Status 200",               r.status_code == 200)
    check("threat_score present",     "threat_score"  in r.json())
    check("is_attack present",        "is_attack"     in r.json())
    check("threat_level present",     "threat_level"  in r.json())
    check("model_scores present",     "model_scores"  in r.json())
    check("4 model scores returned",  len(r.json().get("model_scores", {})) == 4)
    check("score is float 0-1",       0.0 <= r.json()["threat_score"] <= 1.0)

    print("\nPOST /predict  (high-value vector — may score higher)")
    r = requests.post(f"{BASE}/predict", json={"features": highs})
    check("Status 200",         r.status_code == 200)
    check("score is float 0-1", 0.0 <= r.json()["threat_score"] <= 1.0)

    # ── Wrong feature count ───────────────────────────────────────────────────
    print("\nPOST /predict  (wrong feature count — expect 422)")
    r = requests.post(f"{BASE}/predict", json={"features": [0.0, 1.0, 2.0]})
    check("Status 422",  r.status_code == 422)

    # ── Batch prediction ──────────────────────────────────────────────────────
    print("\nPOST /predict/batch  (5 flows)")
    batch = {"flows": [zeros] * 3 + [highs] * 2}
    r = requests.post(f"{BASE}/predict/batch", json=batch)
    check("Status 200",              r.status_code == 200)
    check("flow_count = 5",          r.json().get("flow_count")   == 5)
    check("5 predictions returned",  len(r.json().get("predictions", [])) == 5)
    check("attack_rate present",     "attack_rate" in r.json())

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"Results:  {PASS} passed  |  {FAIL} failed")
    print(f"{'='*45}\n")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    try:
        requests.get(f"{BASE}/health", timeout=3)
    except Exception:
        print(f"ERROR: Server not running at {BASE}")
        print("Start it first with:")
        print("  uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    run_tests()