# TremorTrace Integration Test

## Single End-to-End Test

This test verifies the complete frontend-backend-ML pipeline works correctly.

### Prerequisites

1. **Backend models ready**: Ensure ML models exist at:
   - `backend/models/spiral_final.keras`
   - `backend/models/wave_final.keras`
   - `backend/metadata.json`

2. **Environment configured**: `.env.local` exists with `NEXT_PUBLIC_API_URL=http://localhost:8000`

---

## Test Procedure

### Step 1: Start Backend Server

```bash
# Terminal 1 - Backend
cd backend
python -m uvicorn app:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
✓ Models loaded successfully
```

**Verify Backend Health:**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.1.0"
}
```

---

### Step 2: Start Frontend Server

```bash
# Terminal 2 - Frontend
npm run dev
```

**Expected Output:**
```
▲ Next.js 14.2.35
- Local:        http://localhost:3000
✓ Ready in 2.3s
```

---

### Step 3: Complete Prediction Workflow

1. **Open Browser:**
   - Navigate to: http://localhost:3000/detector

2. **Verify Health Check:**
   - Page loads without "Backend Unavailable" error banner
   - No console errors in DevTools (F12)

3. **Test Spiral Drawing:**
   - Click on "Archimedes Spiral" tab (should be active by default)
   - Draw a spiral pattern on the canvas (3-4 rotations from center outward)
   - Observe:
     - ✅ "Recording" indicator appears while drawing
     - ✅ Metrics update in right sidebar (Tremor Score, Radial Deviation, etc.)

4. **Test Wave Drawing:**
   - Click "Wave Analysis" tab
   - Draw a wave pattern left-to-right following the guide
   - Observe:
     - ✅ "Recording" indicator appears
     - ✅ Metrics update for wave test

5. **Run Prediction:**
   - Click the "Predict" button (bottom right)
   - Observe:
     - ✅ Button shows "Predicting…" with spinner
     - ✅ Network request to `http://localhost:8000/predict` (visible in DevTools Network tab)
     - ✅ Response received in ~2-3 seconds

6. **Verify Results Display:**
   - **PD Probability Percentage** displays (0-100%)
   - **Risk Tier Badge** shows with colored background (Low/Mild/Moderate/Elevated/High Risk)
   - **Progress Bar** fills to percentage
   - **Model Breakdown** shows:
     - Spiral CNN: XX%
     - Wave CNN: XX%
   - **Confidence Score** displays with label (Low/Moderate/High)
   - **Model Agreement** shows "Unanimous" or "Partial"
   - **Medical Disclaimer** appears at bottom

7. **Verify Grad-CAM Heatmaps:**
   - Scroll down to "Model Explainability" section
   - ✅ Two heatmap images visible (Spiral and Wave)
   - ✅ Red/yellow regions highlight areas of tremor/irregularity
   - ✅ Blue regions on smooth areas
   - Click "Heatmap"/"Original" toggle buttons
   - ✅ Images switch between heatmap overlay and original drawing

---

## Success Criteria

✅ **Backend:** Models load, health check passes, prediction endpoint responds
✅ **Frontend:** Page loads, canvases work, results display
✅ **Integration:** Request/response cycle completes successfully
✅ **ML Pipeline:** Dual CNN predictions, Grad-CAM heatmaps generated
✅ **Data Flow:** All response fields displayed correctly in UI

---

## Quick Verification Checklist

- [ ] Backend health check returns `"models_loaded": true`
- [ ] Frontend detector page loads without errors
- [ ] Can draw on both spiral and wave canvases
- [ ] "Predict" button triggers network request
- [ ] Prediction displays with percentage, risk tier, and model breakdown
- [ ] Grad-CAM heatmaps appear and toggle works
- [ ] No console errors in browser DevTools
- [ ] Backend terminal shows successful `/predict` POST request

---

## Troubleshooting

**Issue:** Backend shows "Models not found"
- **Fix:** Run training script first: `python ml/train_local.py`

**Issue:** Frontend shows "Backend Unavailable"
- **Fix:** Verify backend is running on port 8000, check `.env.local` has correct URL

**Issue:** CORS errors in browser
- **Fix:** Backend already has CORS enabled for all origins (line 49-55 in `app.py`)

**Issue:** Prediction takes > 10 seconds
- **Fix:** Normal on first request (TensorFlow compilation), subsequent requests ~2s

---

## Test Data Locations

If you want to test file upload mode instead of drawing:

**Spiral Test Images:**
`ml/src/data/drawings/spiral/testing/parkinson/*.png`

**Wave Test Images:**
`ml/src/data/drawings/wave/testing/parkinson/*.png`

Upload both, click Predict, verify same workflow.

---

## Expected Network Request/Response

**Request to `POST /predict`:**
```json
{
  "spiral_image": "data:image/png;base64,iVBOR...",
  "wave_image": "data:image/png;base64,iVBOR...",
  "input_mode": "drawn"
}
```

**Response (example):**
```json
{
  "pd_probability_percent": 67.45,
  "risk_tier": "Moderate Risk",
  "risk_color": "#E67E22",
  "spiral_cnn_percent": 62.34,
  "wave_cnn_percent": 70.12,
  "confidence_score": 0.349,
  "confidence_label": "Moderate",
  "spiral_gradcam_base64": "data:image/png;base64,...",
  "wave_gradcam_base64": "data:image/png;base64,...",
  "model_agreement": 0.5,
  "unanimous": false,
  "disclaimer": "This is a screening tool..."
}
```

---

**Test Duration:** ~2-3 minutes
**Automation:** Manual (visual verification required for UI elements)
