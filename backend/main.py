"""
TremorTrace — FastAPI backend.
Receives a base64-encoded spiral image + stroke points,
runs MobileNetV2 CNN prediction, generates Grad-CAM and speed heatmaps.
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import base64, os, traceback
import numpy as np
from io import BytesIO
from PIL import Image

from models.cnn_model import load_cnn_model, predict_cnn
from models.gradcam import generate_gradcam
from utils.image_processing import (
    decode_base64_image, preprocess_for_model, image_to_base64,
)
from utils.stroke_analysis import generate_speed_heatmap

app = FastAPI(title="TremorTrace API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model reference (loaded once at startup) ──────────
cnn_model = None


@app.on_event("startup")
async def startup_load_models():
    global cnn_model
    cnn_model = load_cnn_model()
    print("Models loaded successfully")


# ── Request / Response schemas ────────────────────────────────
class StrokePoint(BaseModel):
    x: float
    y: float
    t: float


class AnalyzeRequest(BaseModel):
    image: str  # Base64-encoded PNG from canvas (data:image/png;base64,…)
    strokes: List[StrokePoint]


class AnalyzeResponse(BaseModel):
    prediction: str
    probability: float
    confidence: float
    cnn_probability: float
    label: str
    heatmap: str           # Base64 Grad-CAM overlay
    speed_heatmap: str     # Base64 speed visualisation


# ── Routes ────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_spiral(request: AnalyzeRequest):
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. Decode & preprocess (grayscale → 3-channel)
        pil_image = decode_base64_image(request.image)
        image_array = preprocess_for_model(pil_image)  # (224,224,3) float32 [0,1]

        # 2. CNN prediction
        cnn_prob = predict_cnn(cnn_model, image_array)

        # 3. Stroke data for speed heatmap
        strokes_dicts = [{"x": s.x, "y": s.y, "t": s.t} for s in request.strokes]

        # 4. Determine final result
        probability = cnn_prob
        threshold = 0.5
        is_parkinson = probability >= threshold
        prediction = "parkinson" if is_parkinson else "healthy"
        confidence = abs(probability - 0.5) * 2  # 0→1 scale

        label_map = {
            "parkinson": "Indicators of Parkinson's tremor detected",
            "healthy": "Drawing appears within healthy range",
        }

        # 5. Grad-CAM
        _, overlay = generate_gradcam(cnn_model, image_array)
        overlay_b64 = image_to_base64(overlay)

        # 6. Speed heatmap (visual only — not used for prediction)
        speed_heatmap_b64 = generate_speed_heatmap(strokes_dicts, pil_image.size)

        return AnalyzeResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=round(confidence, 4),
            cnn_probability=round(cnn_prob, 4),
            label=label_map[prediction],
            heatmap=overlay_b64,
            speed_heatmap=speed_heatmap_b64,
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Analyse an uploaded image file directly (no strokes)."""
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents))
        image_array = preprocess_for_model(pil_image)

        cnn_prob = predict_cnn(cnn_model, image_array)
        probability = cnn_prob
        threshold = 0.5
        is_parkinson = probability >= threshold
        prediction = "parkinson" if is_parkinson else "healthy"
        confidence = abs(probability - 0.5) * 2

        label_map = {
            "parkinson": "Indicators of Parkinson's tremor detected",
            "healthy": "Drawing appears within healthy range",
        }

        _, overlay = generate_gradcam(cnn_model, image_array)
        overlay_b64 = image_to_base64(overlay)

        # No strokes for uploaded image — blank speed heatmap
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        speed_heatmap_b64 = image_to_base64(blank)

        return AnalyzeResponse(
            prediction=prediction,
            probability=round(probability, 4),
            confidence=round(confidence, 4),
            cnn_probability=round(cnn_prob, 4),
            label=label_map[prediction],
            heatmap=overlay_b64,
            speed_heatmap=speed_heatmap_b64,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": cnn_model is not None}


# ── Serve built React frontend in production ──────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(STATIC_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
