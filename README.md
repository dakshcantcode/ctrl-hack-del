# TremorTrace

**Parkinson's tremor screening via spiral drawing analysis.**

Draw a spiral on the canvas → the app analyses it using a MobileNetV2 CNN (trained on grayscale spiral images) and kinematic stroke features, then returns:
- **Prediction** (healthy / parkinson indicators) with probability
- **Grad-CAM heatmap** showing which regions influenced the model
- **Speed heatmap** showing drawing steadiness (green = steady, red = erratic)
- **Kinematic features** extracted from your pen strokes

---

## Project structure

```
yorkhacks26/
├── training/               # ML training pipeline
│   ├── config.py           # Hyperparameters & paths
│   ├── dataset.py          # Load images, augmentation
│   ├── model.py            # MobileNetV2 architecture
│   ├── train.py            # Run training (Phase 1 frozen → Phase 2 fine-tune)
│   └── data/
│       ├── healthy/        # ← Put healthy spiral images here
│       └── parkinson/      # ← Put parkinson spiral images here
│
├── backend/                # FastAPI server
│   ├── main.py             # Routes, CORS, model loading
│   ├── models/
│   │   ├── cnn_model.py    # Load & predict with MobileNetV2
│   │   └── gradcam.py      # Grad-CAM generation
│   ├── utils/
│   │   ├── image_processing.py  # base64 ↔ image, grayscale conversion
│   │   └── stroke_analysis.py   # Kinematic features, speed heatmap
│   ├── saved_models/       # ← spiral_cnn.h5 goes here (auto-copied by train.py)
│   └── requirements.txt
│
├── frontend/               # React + Vite
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SpiralCanvas.jsx
│   │   │   └── Results.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── .replit                 # Replit deployment config
├── replit.nix              # Nix packages
├── run.sh                  # Single startup script
└── README.md
```

## Quick start (local)

### 1. Train the model
```bash
cd training
pip install -r requirements.txt
# Place images in data/healthy/ and data/parkinson/
python train.py
```

### 2. Run the backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — draw a spiral and click **Analyse**.

## Replit deployment

1. Push this repo to Replit (or import from GitHub).
2. The `.replit` file auto-runs `run.sh` which installs deps, builds the frontend, and starts FastAPI.
3. Deploy via **Replit Deployments → Cloud Run**.

## Key design decisions

| Decision | Rationale |
|---|---|
| **Grayscale focus** | Spirals are structural — colour carries no diagnostic info. Converting to gray removes noise. Replicated to 3ch for MobileNetV2 compatibility. |
| **No flip/shear augmentation** | Flips change spiral chirality; shear distorts the tremor signal. |
| **360° rotation** | Spirals are rotationally invariant — free augmentation. |
| **Dummy model fallback** | Server starts even without `spiral_cnn.h5` for development. |
