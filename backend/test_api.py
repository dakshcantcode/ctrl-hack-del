#!/usr/bin/env python3
"""
TremorTrace API — Test Script

Tests all endpoints against a running server.

Usage:
    # Start server first:  uvicorn app:app --reload
    # Then run:            python test_api.py
"""

import os
import sys
import base64
import json
import urllib.request
import urllib.error

BASE_URL = os.environ.get("API_URL", "http://localhost:8000")

# Find a test image relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SPIRAL_TEST = os.path.join(PROJECT_ROOT, "ml", "src", "data", "drawings", "spiral", "testing", "parkinson")
WAVE_TEST = os.path.join(PROJECT_ROOT, "ml", "src", "data", "drawings", "wave", "testing", "parkinson")


def api_get(path):
    """GET request to the API."""
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def api_post(path, data):
    """POST JSON request to the API."""
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_health():
    """Test GET /health."""
    print("1. GET /health")
    result = api_get("/health")
    assert result["status"] == "healthy", f"Expected healthy, got {result['status']}"
    assert result["models_loaded"] is True, "Models not loaded"
    print(f"   OK: status={result['status']}, version={result.get('version')}")


def test_model_info():
    """Test GET /model-info."""
    print("2. GET /model-info")
    result = api_get("/model-info")
    assert "models" in result, "Missing 'models' key"
    assert "spiral_cnn" in result["models"], "Missing spiral_cnn"
    assert "wave_cnn" in result["models"], "Missing wave_cnn"
    print(f"   OK: version={result.get('version')}, spiral_acc={result['models']['spiral_cnn']['accuracy']}")


def test_predict():
    """Test POST /predict with real test images."""
    print("3. POST /predict (real images)")

    # Load test images
    spiral_files = sorted([f for f in os.listdir(SPIRAL_TEST) if f.endswith(".png")])
    wave_files = sorted([f for f in os.listdir(WAVE_TEST) if f.endswith(".png")])

    if not spiral_files or not wave_files:
        print("   SKIP: No test images found")
        return

    with open(os.path.join(SPIRAL_TEST, spiral_files[0]), "rb") as f:
        spiral_b64 = base64.b64encode(f.read()).decode()
    with open(os.path.join(WAVE_TEST, wave_files[0]), "rb") as f:
        wave_b64 = base64.b64encode(f.read()).decode()

    status, result = api_post("/predict", {
        "spiral_image": spiral_b64,
        "wave_image": wave_b64,
        "input_mode": "drawn",
    })

    assert status == 200, f"Expected 200, got {status}: {result}"

    # Validate all required keys
    required_keys = [
        "pd_probability_percent", "risk_tier", "risk_color",
        "spiral_cnn_percent", "wave_cnn_percent", "input_mode",
        "weights_used", "model_agreement", "unanimous",
        "confidence_score", "confidence_label", "disclaimer",
        "spiral_gradcam_base64", "wave_gradcam_base64",
    ]
    missing = [k for k in required_keys if k not in result]
    assert not missing, f"Missing keys: {missing}"

    print(f"   OK: PD={result['pd_probability_percent']}%, tier={result['risk_tier']}")
    print(f"       spiral={result['spiral_cnn_percent']}%, wave={result['wave_cnn_percent']}%")
    print(f"       gradcam_spiral={'present' if result['spiral_gradcam_base64'] else 'MISSING'}")
    print(f"       gradcam_wave={'present' if result['wave_gradcam_base64'] else 'MISSING'}")


def test_predict_with_data_uri():
    """Test POST /predict with data URI prefix."""
    print("4. POST /predict (data URI prefix)")

    spiral_files = sorted([f for f in os.listdir(SPIRAL_TEST) if f.endswith(".png")])
    wave_files = sorted([f for f in os.listdir(WAVE_TEST) if f.endswith(".png")])

    with open(os.path.join(SPIRAL_TEST, spiral_files[0]), "rb") as f:
        spiral_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    with open(os.path.join(WAVE_TEST, wave_files[0]), "rb") as f:
        wave_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    status, result = api_post("/predict", {
        "spiral_image": spiral_b64,
        "wave_image": wave_b64,
        "input_mode": "uploaded",
    })

    assert status == 200, f"Expected 200, got {status}: {result}"
    print(f"   OK: PD={result['pd_probability_percent']}%, mode={result['input_mode']}")


def test_predict_validation():
    """Test POST /predict with invalid input."""
    print("5. POST /predict (validation errors)")

    # Missing fields
    status, result = api_post("/predict", {})
    assert status == 422, f"Expected 422, got {status}"
    print(f"   OK: empty body → {status}")

    # Invalid base64
    status, result = api_post("/predict", {
        "spiral_image": "not-valid-base64!!!",
        "wave_image": "also-not-valid!!!",
        "input_mode": "drawn",
    })
    assert status == 400, f"Expected 400, got {status}"
    print(f"   OK: invalid base64 → {status}")


def main():
    print("=" * 50)
    print("  TremorTrace API Tests")
    print(f"  Server: {BASE_URL}")
    print("=" * 50)
    print()

    try:
        test_health()
        test_model_info()
        test_predict()
        test_predict_with_data_uri()
        test_predict_validation()
    except Exception as e:
        print(f"\n   FAILED: {e}")
        sys.exit(1)

    print()
    print("=" * 50)
    print("  ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
