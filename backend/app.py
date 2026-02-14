"""
TremorTrace — FastAPI Backend

Serves Parkinson's disease screening predictions via REST API.
Loads two MobileNetV2 CNN models (spiral + wave) at startup and
runs inference on base64-encoded drawing images.

Endpoints:
    POST /predict    — main prediction (both images → probability + Grad-CAM)
    GET  /health     — health check (models loaded?)
    GET  /model-info — model metadata and accuracy metrics
"""

import os
import json
import time
import logging
import warnings

# Suppress TF warnings for cleaner server logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ML pipeline imports (self-contained files in ml/)
from ml.input_handler import process_input

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

# =============================================================================
# App Setup
# =============================================================================
app = FastAPI(
    title="TremorTrace API",
    description="Parkinson's disease screening from spiral and wave drawings",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State
# =============================================================================
spiral_model = None
wave_model = None
metadata = None
models_loaded = False

logger = logging.getLogger("tremortrace")
logging.basicConfig(level=logging.INFO)


# =============================================================================
# Startup
# =============================================================================
@app.on_event("startup")
async def load_models():
    """Load models into memory and run warm-up inference."""
    global spiral_model, wave_model, metadata, models_loaded

    logger.info("Loading models...")
    start = time.time()

    spiral_path = os.path.join(MODELS_DIR, "spiral_final.keras")
    wave_path = os.path.join(MODELS_DIR, "wave_final.keras")

    spiral_model = tf.keras.models.load_model(spiral_path)
    wave_model = tf.keras.models.load_model(wave_path)

    # Load metadata
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    load_time = time.time() - start
    logger.info(f"Models loaded in {load_time:.1f}s")

    # Warm-up: run dummy inference to pre-compile TF graph
    logger.info("Running warm-up inference...")
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    spiral_model.predict(dummy, verbose=0)
    wave_model.predict(dummy, verbose=0)
    logger.info("Warm-up complete. Server ready.")

    models_loaded = True


# =============================================================================
# Request/Response Models
# =============================================================================
class PredictRequest(BaseModel):
    spiral_image: str = Field(..., description="Base64-encoded spiral drawing (with or without data URI prefix)")
    wave_image: str = Field(..., description="Base64-encoded wave drawing (with or without data URI prefix)")
    input_mode: str = Field(default="drawn", description="'drawn' or 'uploaded' (metadata only)")


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health")
async def health_check():
    """Health check — is the server up and are models loaded?"""
    return {
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "version": metadata.get("version", "unknown") if metadata else "unknown",
    }


@app.get("/model-info")
async def model_info():
    """Returns model metadata (accuracy, architecture, weights, etc.)."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded yet")
    return metadata


@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Main prediction endpoint.

    Accepts base64-encoded spiral and wave drawings, runs both CNNs,
    combines via weighted ensemble, generates Grad-CAM heatmaps,
    and returns the full result.
    """
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading. Try again in a few seconds.")

    # Validate input_mode
    if request.input_mode not in ("drawn", "uploaded"):
        raise HTTPException(status_code=400, detail="input_mode must be 'drawn' or 'uploaded'")

    # Run the ML pipeline
    result = process_input(
        spiral_image_base64=request.spiral_image,
        wave_image_base64=request.wave_image,
        spiral_cnn_model=spiral_model,
        wave_cnn_model=wave_model,
        input_mode=request.input_mode,
    )

    # Check for ML pipeline errors
    if "error" in result:
        raise HTTPException(
            status_code=400,
            detail={"error": result["error"], "error_type": result["error_type"]},
        )

    return result
