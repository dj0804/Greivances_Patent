#!/usr/bin/env bash

# Setup script to train and prepare the model for the API
# This fixes the 503 "Model not found" error

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
BACKEND_PYTHON="$BACKEND_DIR/venv/bin/python"

# Use venv python if available, otherwise system python
if [[ -x "$BACKEND_PYTHON" ]]; then
  PYTHON="$BACKEND_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Error: Python not found"
  exit 1
fi

cd "$ROOT_DIR"

echo "🔧 Model Setup for Hostel Grievance API"
echo "========================================"
echo ""

echo "📝 Step 1: Generating 500 mock training samples..."
"$PYTHON" - << 'PYEOF'
import sys
from pathlib import Path
backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))
from scripts.generate_mock_data import generate_complaints
generate_complaints(500)
PYEOF

echo "✅ Data generated"
echo ""

echo "🤖 Step 2: Training the model (this takes ~2-3 minutes)..."
cd "$BACKEND_DIR"
"$PYTHON" scripts/train_model.py \
  --data data/raw/complaints/mock_training_data.json \
  --model-type cnn_bilstm \
  --output outputs/models/ \
  --epochs 5 \
  --batch-size 32

echo "✅ Training complete"
echo ""

echo "🔗 Step 3: Verifying model setup..."
cd "$ROOT_DIR"
if [[ -f "$BACKEND_DIR/outputs/models/best_model.h5" ]]; then
  SIZE=$(du -h "$BACKEND_DIR/outputs/models/best_model.h5" | cut -f1)
  echo "✅ Model ready: best_model.h5 ($SIZE)"
else
  echo "⚠️  Warning: best_model.h5 not found at expected location"
  echo "   Please check: $BACKEND_DIR/outputs/models/"
  exit 1
fi

echo ""
echo "=========================================="
echo "✨ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the dev server:  ./run_dev.sh"
echo "  2. Visit API docs:      http://localhost:8000/docs"
echo "  3. Test prediction:     http://localhost:3000"
echo ""
