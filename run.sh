#!/usr/bin/env bash
set -e

echo "══════════════════════════════════════════════"
echo " TremorTrace — starting up"
echo "══════════════════════════════════════════════"

# ── 1. Install Python deps ──────────────────────────────────
echo "[1/4] Installing Python dependencies…"
pip install -q -r backend/requirements.txt

# ── 2. Build React frontend ─────────────────────────────────
echo "[2/4] Building frontend…"
cd frontend
npm install --silent
npm run build
cd ..

# ── 3. Copy built frontend into backend static serving ──────
echo "[3/4] Copying frontend build to backend…"
rm -rf backend/static
cp -r frontend/dist backend/static

# ── 4. Start FastAPI (serves API + static frontend) ─────────
echo "[4/4] Starting FastAPI server on port ${PORT:-8000}…"
cd backend
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
