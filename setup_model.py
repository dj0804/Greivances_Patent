#!/usr/bin/env python3
"""
Setup script to train and prepare the model for the API.
Fixes the 503 "Model not found" error by:
1. Installing dependencies
2. Generating mock training data
3. Training the model
4. Ensuring it's saved with the correct name
"""

import sys
import os
from pathlib import Path
import subprocess

# Add backend to path
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"

def find_python_executable():
    """Find the correct Python executable to use."""
    # Priority: venv python > system python3 > system python
    candidates = [
        BACKEND_DIR / "venv" / "bin" / "python",
        BACKEND_DIR / "venv" / "Scripts" / "python.exe",  # Windows
        Path("/opt/homebrew/bin/python3"),  # Apple Silicon
        Path("/usr/local/bin/python3"),
        Path("/usr/bin/python3"),
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    
    # Fallback to sys.executable
    return sys.executable

def run_command(cmd, description, use_venv_python=False):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    if use_venv_python:
        python_exe = find_python_executable()
        cmd = cmd.replace("python", python_exe, 1)
        print(f"   Using Python: {python_exe}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Complete: {description}")

def main():
    os.chdir(PROJECT_ROOT)
    python_exe = find_python_executable()
    
    print("🔧 Model Setup for Hostel Grievance API")
    print("=" * 60)
    print(f"Using Python: {python_exe}")
    print(f"Working directory: {PROJECT_ROOT}")
    print()
    
    # Step 0: Install dependencies
    print("📦 Step 0: Installing backend dependencies...")
    install_cmd = f"{python_exe} -m pip install -q -r backend/requirements.txt"
    result = subprocess.run(install_cmd, shell=True)
    if result.returncode != 0:
        print("⚠️  Warning: Some dependencies may not have installed properly")
        print("   Continuing anyway...")
    else:
        print("✅ Dependencies installed")
    
    print()
    os.chdir(BACKEND_DIR)
    
    # Step 1: Generate mock data
    print("📝 Step 1: Generating mock training data (500 samples)...")
    try:
        from scripts.generate_mock_data import generate_complaints
        generate_complaints(500)  # 500 samples for faster training
        print("✅ Data generated")
    except Exception as e:
        print(f"❌ Failed to generate data: {e}")
        sys.exit(1)
    
    # Step 2: Train the model
    print("\n🤖 Step 2: Training the model (this takes ~2-3 minutes on CPU)...")
    train_cmd = (
        f"{python_exe} scripts/train_model.py "
        f"--data data/raw/complaints/mock_training_data.json "
        f"--model-type cnn_bilstm "
        f"--output outputs/models/ "
        f"--epochs 5 "
        f"--batch-size 32"
    )
    
    result = subprocess.run(train_cmd, shell=True)
    if result.returncode != 0:
        print("❌ Model training failed")
        sys.exit(1)
    print("✅ Model training complete")
    
    # Step 3: Verify model was created
    print("\n🔗 Step 3: Verifying model setup...")
    models_dir = Path("outputs/models")
    expected_model = models_dir / "best_model.h5"
    
    if expected_model.exists():
        size_mb = expected_model.stat().st_size / 1024 / 1024
        print(f"✅ Model found: best_model.h5 ({size_mb:.1f} MB)")
    else:
        # Check for backup name
        backup_models = list(models_dir.glob("model_*_final.h5"))
        if backup_models:
            print(f"⚠️  Found backup model: {backup_models[0].name}")
            print(f"   Copying to expected location...")
            import shutil
            shutil.copy(backup_models[0], expected_model)
            size_mb = expected_model.stat().st_size / 1024 / 1024
            print(f"✅ Model verified: best_model.h5 ({size_mb:.1f} MB)")
        else:
            print("❌ No model file found. Training may have failed.")
            print(f"   Checked location: {models_dir}")
            sys.exit(1)
    
    # Step 4: Verify required files
    print("\n📦 Step 4: Verifying required files...")
    required_files = [
        models_dir / "best_model.h5",
        models_dir / "tokenizer.pkl",
    ]
    
    for file in required_files:
        if file.exists():
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  ✅ {file.name:<25} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  {file.name:<25} (optional, may not be required)")
    
    os.chdir(PROJECT_ROOT)
    print("\n" + "="*60)
    print("✨ Setup complete!")
    print("="*60)
    print("\nYou can now start the dev server:")
    print("  ./run_dev.sh")
    print("\nThe API will be available at:")
    print("  http://localhost:8000/docs  (API documentation)")
    print("  http://localhost:3000       (Frontend)")
    print("  http://localhost:8000/api/v1/predict  (prediction endpoint)")
    print()

if __name__ == "__main__":
    main()
