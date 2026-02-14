import base64
import cv2
import numpy as np
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- 1. INITIALIZATION ---
app = FastAPI(title="PD Detection System")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 2. DATA SCHEMAS ---
class StrokePoint(BaseModel):
    x: float
    y: float
    t: float


class AnalysisRequest(BaseModel):
    image: str  # Base64 string
    strokes: List[StrokePoint]


# --- 3. FEATURE ENGINEERING (XGBoost Path) ---
def extract_kinematics(strokes: List[StrokePoint]):
    """Calculates velocity and acceleration for the Feature Model."""
    if len(strokes) < 2:
        return np.zeros((1, 3))

    velocities = []
    accelerations = []

    for i in range(1, len(strokes)):
        dist = math.sqrt((strokes[i].x - strokes[i - 1].x) ** 2 + (strokes[i].y - strokes[i - 1].y) ** 2)
        dt = (strokes[i].t - strokes[i - 1].t) or 0.001

        v = dist / dt
        velocities.append(v)

        if len(velocities) > 1:
            accel = (velocities[-1] - velocities[-2]) / dt
            accelerations.append(accel)

    avg_v = np.mean(velocities)
    max_a = np.max(accelerations) if accelerations else 0
    total_time = strokes[-1].t - strokes[0].t

    return np.array([[avg_v, max_a, total_time]])


# --- 4. ROBUST IMAGE PREPROCESSING (CNN Path) ---
def process_base64_image(base64_str: str):
    """Fail-proof decoding of Base64 to OpenCV image."""
    try:
        # A. Strip potential Data URI header
        if "," in base64_str:
            base64_data = base64_str.split(",")[1]
        else:
            base64_data = base64_str

        # B. Fix missing padding (Base64 must be multiple of 4)
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += "=" * (4 - missing_padding)

        # C. Decode to bytes and convert to numpy array
        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)

        # D. Decode with OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("OpenCV returned None. The string is not a valid image format.")

        # E. Standardize for MobileNetV2
        img = cv2.resize(img, (224, 224))
        return img.astype('float32') / 255.0

    except Exception as e:
        print(f"DEBUG: Image Processing Error -> {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


# --- 5. MAIN ENDPOINT ---
@app.post("/analyze")
async def analyze_drawing(request: AnalysisRequest):
    # Validation
    if not request.strokes:
        raise HTTPException(status_code=400, detail="No stroke data provided")

    # Image and Feature Processing
    img_input = process_base64_image(request.image)
    kinematic_features = extract_kinematics(request.strokes)

    # Mock Model Outputs (Replace with model.predict() later)
    p_cnn = 0.65
    p_xgb = 0.72

    # Ensemble logic: 40% Image, 60% Kinematics
    ensemble_p = (0.4 * p_cnn) + (0.6 * p_xgb)

    return {
        "prediction": "Parkinson's Likely" if ensemble_p > 0.5 else "Healthy Control",
        "probabilities": {
            "cnn_score": round(p_cnn, 4),
            "kinematic_score": round(p_xgb, 4),
            "ensemble_final": round(ensemble_p, 4)
        },
        "meta": {
            "points_count": len(request.strokes),
            "duration": round(request.strokes[-1].t - request.strokes[0].t, 2)
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)