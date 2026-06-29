# Hostel Grievance Urgency Classification & Routing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

**An intelligent multi-stage decision pipeline for automated complaint prioritization combining deep learning, semantic vector search, temporal pattern analysis, and density-based clustering.**

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Features](#system-features)
- [Architecture Summary](#architecture-summary)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [API Usage](#api-usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Hostel Grievance Urgency Classification & Routing System** addresses the challenge of triaging large volumes of complaints in institutional hostel environments. Unlike traditional text classifiers that operate on content alone, this system implements a **hybrid multi-signal architecture** that jointly analyzes:

1. **Semantic Similarity** (transformer-based embeddings + vector database)
2. **Inferred Urgency** (CNN-BiLSTM deep learning model)
3. **Temporal Frequency Patterns** (JSON-based temporal encoding with burst detection)
4. **Historical Resolution Feedback** (self-calibrating urgency thresholds)
5. **Cluster Density** (HDBSCAN clustering over embeddings for systemic issue detection)

The system is **not a simple classifier**—it is a **multi-stage decision pipeline** with modular components that produce context-aware, adaptive priority scores suitable for real-world deployment in high-volume complaint management scenarios.

---

## Problem Statement

### Challenges in Manual Complaint Management

Institutional hostels receive **hundreds to thousands** of complaints daily across categories:
- **Maintenance**: Water heaters, ACs, electrical issues
- **Safety**: Door locks, lighting, security concerns
- **Cleanliness**: Room cleaning, common area maintenance
- **Food**: Dining hall quality, hygiene issues
- **Administrative**: Fee issues, room allocation

**Problems with Manual Triage**:
1. **High Volume**: Staff cannot read and prioritize 500+ daily complaints
2. **Context Blindness**: A single "AC broken" complaint may be low urgency, but 30 similar complaints indicate a systemic HVAC failure
3. **Time Sensitivity**: A door lock complaint at 2 AM is far more urgent than the same complaint at 2 PM
4. **Subjective Judgment**: Different staff members prioritize differently, causing inconsistency
5. **Delayed Response**: Critical issues buried in noise lead to escalation and student dissatisfaction

### Why Existing Solutions Fall Short

| Approach | Limitation |
|----------|------------|
| **Keyword Matching** | Easily gamed; misses context |
| **Rule-Based Systems** | Brittle; requires constant manual updates |
| **Pure ML Classifiers** | Context-blind; no systemic issue detection |
| **Sentiment Analysis** | Urgency ≠ sentiment; misses technical failures |

**Our Solution**: A **hybrid decision architecture** that combines strengths of ML, vector search, temporal analysis, and rule-based logic to produce robust, explainable priority scores.

---

## System Features

### Core Capabilities

✅ **Intelligent Preprocessing**
- Emotion-preserving text cleaning (retains ALL CAPS, excessive punctuation, emojis as urgency signals)
- Urgency-aware tokenization (preserves negations: "not", "no", "never")
- Temporal feature extraction (hour, day, peak times, frequency patterns)

✅ **Hybrid Deep Learning Architecture**
- **CNN Layer**: Extracts local n-gram patterns indicative of urgency
- **BiLSTM Layer**: Captures long-range contextual dependencies
- **Dense Head**: Outputs urgency probability distribution

✅ **Semantic Vector Search**
- Transformer-based sentence embeddings (Sentence-BERT / Universal Sentence Encoder)
- Vector database for fast similarity search
- Identifies similar complaints for cluster analysis

✅ **Density-Based Clustering**
- HDBSCAN automatic clustering (no manual cluster count)
- Spatial density: Semantic similarity within clusters
- Temporal density: Complaint frequency over time
- **Systemic Issue Detection**: High-density clusters indicate infrastructure failures

✅ **Temporal Frequency Analysis**
- JSON-based encoding of time features
- Burst detection (2x spike in complaint rate → outbreak)
- Time-of-day adjustments (night, peak hours, weekends)

✅ **Historical Feedback Loop**
- Tracks actual resolution times vs. predicted urgency
- Self-calibrating weights based on SLA performance
- **No retraining required** for adaptation

✅ **Multi-Signal Decision Fusion**
- Weighted aggregation of 5 signals (model, cluster, temporal, history, semantic)
- Adaptive weight tuning based on system state
- Override rules for safety-critical edge cases

✅ **RESTful API & Visual Intelligence Frontend**
- FastAPI endpoint for real-time predictions
- Next.js Admin Dashboard with Recharts visualization
- Live Heatmaps & Systemic Cluster Trends
- Real-time SLA breach indicators

✅ **Intelligent Summarization**
- HuggingFace Transformers integration (`distilbart-cnn-12-6`)
- Auto-extracts concise summaries from long grievances
- Runs concurrently without blocking the core AI inference layer

---

## Architecture Summary

### High-Level Flow

```
Raw Complaint + Metadata
        ↓
┌───────────────────┐
│  PREPROCESSING     │
│  • Clean Text      │
│  • Tokenize        │
│  • Temporal Extract│
└─────────┬─────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
┌────────┐  ┌──────────┐
│BRANCH A│  │ BRANCH B │
│CNN-BiLSTM│ │ Semantic │
│Urgency │  │Embeddings│
│Predict │  │+ Cluster │
└────┬───┘  └────┬─────┘
     │           │
     └─────┬─────┘
           ▼
    ┌──────────────┐
    │DECISION ENGINE│
    │Multi-Signal  │
    │Fusion        │
    └──────┬───────┘
           ▼
    Priority + Routing
```

**Parallel Processing**: Branch A (CNN-BiLSTM) and Branch B (Embeddings + Clustering) run concurrently for 1.5x speedup.

### Decision Fusion Formula

```
Priority = α·U_model + β·D_cluster + γ·T_temporal + δ·H_history + ε·S_semantic

Default Weights:
  α = 0.40  (Model Urgency)
  β = 0.25  (Cluster Density - systemic issues)
  γ = 0.15  (Temporal Context)
  δ = 0.10  (Historical Calibration)
  ε = 0.10  (Semantic Similarity to Critical Cases)
```

**See [docs/architecture.md](docs/architecture.md) for detailed component architecture.**

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow/Keras 2.10+ |
| **Word Embeddings** | FastText (300-dim pretrained) |
| **Sentence Embeddings** | Sentence-BERT / Universal Sentence Encoder |
| **NLP Processing** | spaCy, NLTK, emoji, contractions |
| **Clustering** | HDBSCAN (density-based) |
| **Vector Database** | FAISS |
| **API Framework** | FastAPI |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Serialization** | Pickle, JSON |

**System Requirements**:
- Python 3.8+
- 8GB RAM (16GB recommended for training)
- GPU optional (10x training speedup with CUDA)

---

## Project Structure

```
hostel-patent/
│
├── README.md                          # This file
├── REFACTORING_SUMMARY.md             # Development history
├── requirements.txt                   # Python dependencies
├── run_tests.sh                       # Test automation script
│
├── api/                               # FastAPI application
│   ├── main.py                        # API server entry point
│   ├── routes.py                      # Endpoint definitions
│   └── schemas.py                     # Pydantic request/response models
│
├── data/                              # Data storage (Git LFS recommended)
│   ├── raw/                           # Immutable raw data
│   │   ├── complaints/                # Original complaint texts
│   │   └── metadata/                  # Timestamps, hostel IDs, etc.
│   ├── processed/                     # Transformed artifacts
│   │   ├── tokenized/                 # Clean tokens + vocabulary
│   │   ├── embeddings/                # FastText sequences + sentence embeddings
│   │   └── temporal_json/             # Temporal feature encodings
│   └── external/                      # Pretrained models
│       ├── cc.en.300.bin              # FastText model (download separately)
│       └── sbert_model/               # Sentence-BERT checkpoint
│
├── docs/                              # Comprehensive documentation
│   ├── architecture.md                # System architecture details
│   ├── data_flow.md                   # Data lifecycle & pipeline
│   ├── decision_logic.md              # Decision engine logic (patent-worthy)
│   └── preprocessing.md               # (if exists) Text preprocessing details
│
├── notebooks/                         # Jupyter notebooks for exploration
│
├── outputs/                           # Training and inference outputs
│   ├── models/                        # Trained model checkpoints
│   │   ├── cnn_bilstm_best.h5
│   │   ├── tokenizer.pkl
│   │   └── label_encoder.pkl
│   ├── logs/                          # Training logs & TensorBoard
│   └── reports/                       # Inference predictions, cluster analysis
│
├── scripts/                           # Reproducible execution scripts
│   ├── train_model.py                 # Model training pipeline (JSON/CSV)
│   ├── evaluate_model.py              # Standalone evaluation + metrics report
│   ├── build_embeddings.py            # Vector database construction
│   └── run_inference.py               # Batch/single inference
│
├── src/                               # Core library code
│   ├── __init__.py
│   ├── config.py                      # Centralized configuration
│   │
│   ├── preprocessing/                 # Text preprocessing modules
│   │   ├── __init__.py
│   │   ├── clean_text.py              # Emotion-preserving cleaning
│   │   ├── tokenizer.py               # Urgency-aware tokenization
│   │   └── temporal_encoder.py        # Temporal feature extraction
│   │
│   ├── embeddings/                    # Embedding generation
│   │   ├── __init__.py
│   │   ├── semantic_embedder.py       # FastText encoder
│   │   └── vector_store.py            # Vector database interface
│   │
│   ├── models/                        # Neural network architectures
│   │   ├── __init__.py
│   │   ├── cnn.py                     # CNN model
│   │   ├── bilstm.py                  # BiLSTM model
│   │   ├── cnn_bilstm.py              # Hybrid CNN-BiLSTM
│   │   └── dense_head.py              # Classification head
│   │
│   ├── training/                      # Training pipeline
│   │   ├── __init__.py
│   │   ├── train.py                   # Training logic
│   │   ├── loss.py                    # Custom loss functions
│   │   └── metrics.py                 # Evaluation metrics
│   │
│   ├── inference/                     # Prediction pipeline
│   │   ├── __init__.py
│   │   └── predict.py                 # Inference wrapper
│   │
│   ├── decision_engine/               # Decision fusion layer
│   │   ├── __init__.py
│   │   └── urgency_logic.py           # Multi-signal fusion logic
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── helpers.py                 # General helpers
│       └── logger.py                  # Logging configuration
│
├── frontend/                          # Next.js Visual Intelligence UI
│   ├── src/app/                       # App Router paths
│   │   ├── submit/                    # Student Form (/submit)
│   │   └── admin/                     # Dashboard with Recharts (/admin)
│   └── package.json                   # UI Dependencies
│
└── tests/                             # Unit and integration tests
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_inference.py
```

---

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** or **conda** for package management
- **Git LFS** (optional, for large model files)

### Step 1: Clone Repository

```bash
git clone https://github.com/dj0804/Greivances_Patent.git
cd hostel-patent
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n hostel-patent python=3.8
conda activate hostel-patent
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
- `tensorflow>=2.10.0`
- `spacy`, `nltk`, `gensim`, `fasttext`
- `scikit-learn`, `hdbscan`
- `fastapi`, `pydantic`
- `numpy`, `pandas`, `tqdm`

### Step 4: Download External Models

**spaCy Language Model**:
```bash
python -m spacy download en_core_web_sm
```

**FastText Pretrained Embeddings**:
```bash
# Download from https://fasttext.cc/docs/en/crawl-vectors.html
# Example: English 300-dim
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
mv cc.en.300.bin data/external/
```

**Sentence-BERT Model** (optional, for semantic embeddings):
```python
# Will auto-download on first use via transformers library
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('data/external/sbert_model')
```

### Step 5: Verify Installation

```bash
python -c "import tensorflow as tf; import spacy; print('TensorFlow:', tf.__version__); print('spaCy:', spacy.__version__)"
```

---

## Quick Start

### Run backend + frontend together

From the repository root:

```bash
./run_dev.sh
# or
bash run_dev.sh
```

What it does:
- Starts backend from `backend/` and frontend from `frontend/`
- Avoids port collisions by auto-selecting the next free ports (defaults: backend `8000`, frontend `3000`)
- Connects frontend to backend through frontend API routes using `BACKEND_API_BASE_URL`
- Uses `backend/venv/bin/python` automatically when available
- Ensures FastAPI CLI support is available (`fastapi[standard]`)
- Auto-installs frontend deps if `frontend/node_modules` is missing
- Auto-installs backend deps from `backend/Requirements.txt` if required imports are missing

Optional overrides:

```bash
BACKEND_PORT=8100 FRONTEND_PORT=3100 ./run_dev.sh
```

### 1. Prepare Training Data

The default training corpus is **`support_ticket_train.csv`** at the repository
root — a leakage-free, class-balanced dataset (4,000 train / 1,000 test rows,
33/33/33 across `Low`/`Medium`/`High`). The training script auto-detects file
format by extension, so both CSV and JSON are supported.

**CSV format** (default — `support_ticket_train.csv`):

| Column | Required | Notes |
|--------|----------|-------|
| `Query_Text` | ✅ | Complaint / ticket text |
| `Priority_Label` | ✅ | One of `Low`, `Medium`, `High` |
| `Department` | optional | Read but not used for training (e.g. per-category reporting) |
| `Query_ID` | optional | Row identifier |

The column mapper also accepts `text`/`label` and `Student_Query`/`urgency`
aliases, so most complaint CSVs work without renaming.

**JSON format** (still supported — list of objects):
```json
[
  {
    "id": "C_001",
    "text": "Water heater broken in room 204, no hot water since morning!!!",
    "timestamp": "2026-02-17T08:30:00Z",
    "label": "High",
    "metadata": {"hostel_id": "H-101", "room": "204"}
  },
  {
    "id": "C_002",
    "text": "Light bulb needs replacement in corridor",
    "timestamp": "2026-02-17T10:15:00Z",
    "label": "Low",
    "metadata": {"hostel_id": "H-101", "corridor": "A"}
  }
]
```

### 2. Train the Model

Training uses `support_ticket_train.csv` by default, so `--data` is optional:

```bash
# Uses the default support_ticket_train.csv corpus
python scripts/train_model.py

# Or point at any JSON/CSV file explicitly
python scripts/train_model.py \
    --data path/to/data.csv \
    --model-type cnn_bilstm \
    --epochs 50 \
    --batch-size 32 \
    --output outputs/models/
```

The trainer fits a Keras tokenizer + label encoder, applies **balanced class
weights**, and runs an internal train/validation split.

**Output**:
- `outputs/models/best_model.h5` (trained weights)
- `outputs/models/tokenizer.pkl` (fitted tokenizer — required for inference)
- `outputs/models/label_encoder.pkl` (fitted label encoder)

**Training takes ~30-60 minutes** (GPU recommended).

### 2b. Evaluate the Model

Evaluate against the dedicated held-out test set. Use `--full` so the entire
test file is scored (the data is already split, so no internal slicing):

```bash
python scripts/evaluate_model.py \
    --model outputs/models/best_model.h5 \
    --tokenizer outputs/models/tokenizer.pkl \
    --data support_ticket_test.csv --full \
    --output outputs/reports/evaluation_report.json
```

Reports overall accuracy, macro-F1, per-class precision/recall/F1, and a
confusion matrix (printed and saved as JSON).

### 3. Build Embedding Database

```bash
python scripts/build_embeddings.py \
    --input data/raw/complaints/batch_001.json \
    --model-path data/external/sbert_model \
    --output data/processed/embeddings/vector_store.pkl \
    --run-clustering
```

**Output**:
- `data/processed/embeddings/vector_store.pkl` (searchable vector DB)
- `data/processed/embeddings/cluster_analysis.html` (visualization)

### 4. Run Inference

**Single Complaint**:
```bash
python scripts/run_inference.py \
    --model outputs/models/cnn_bilstm_best.h5 \
    --vector-store data/processed/embeddings/vector_store.pkl \
    --text "The AC in my room has been making loud noises all night, can't sleep!!!" \
    --output outputs/reports/prediction.json
```

**Batch Inference**:
```bash
python scripts/run_inference.py \
    --model outputs/models/cnn_bilstm_best.h5 \
    --vector-store data/processed/embeddings/vector_store.pkl \
    --input data/raw/complaints/new_batch.json \
    --output outputs/reports/batch_predictions.json
```

**Output** (`prediction.json`):
```json
{
  "priority_score": 0.712,
  "urgency_level": "High",
  "routing": "Priority Maintenance Queue",
  "sla_hours": 2,
  "cluster_id": 8,
  "explanation": "High urgency due to model prediction (0.78), night-time filing boost (0.15), and cluster density (0.65 - 12 similar AC complaints this week)"
}
```

### 5. Start API Server

```bash
cd api
fastapi dev --app api.main:app --host 0.0.0.0 --port 8000
```

**API accessible at**: `http://localhost:8000`

**Interactive docs**: `http://localhost:8000/docs`

### 6. Start the Visual Intelligence Frontend

In a new terminal window:

```bash
cd frontend
npm run dev
```

**Frontend available at:**
- Dashboard: `http://localhost:3000/admin`
- Student Form: `http://localhost:3000/submit`

---

## Configuration & Environment Variables

You can adjust standard configurations in `src/config.py`.

### Summarization Module
The new auto-summarization feature uses `sshleifer/distilbart-cnn-12-6` and runs lazily. It can be toggled inside `src/config.py`:
- `ENABLE_SUMMARIZATION = True`
- `SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"`

*Ensure you have `transformers` and `torch` correctly installed as specified in requirements.*

---

## Documentation

Comprehensive technical documentation available in `docs/`:

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | **System Architecture**: Detailed component design, CNN/BiLSTM internals, parallel processing, embedding generation, HDBSCAN clustering |
| [data_flow.md](docs/data_flow.md) | **Data Lifecycle**: Complete data pipeline from raw ingestion to final output, preprocessing steps, embedding construction, inference flow |
| [decision_logic.md](docs/decision_logic.md) | **Decision Engine**: Multi-signal fusion formula, cluster density calculation, temporal frequency analysis, historical calibration, patent-worthy novel aspects |
| [preprocessing.md](docs/preprocessing.md) | **Preprocessing Details**: Text cleaning, tokenization, temporal encoding (if exists) |

**Start Here**: [docs/architecture.md](docs/architecture.md) for system overview.

---

## API Usage

### Endpoint: POST `/predict`

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Water heater broken in room 204, no hot water since morning!!!",
    "timestamp": "2026-02-17T08:30:00Z",
    "metadata": {
      "hostel_id": "H-101",
      "room": "204"
    }
  }'
```

**Response**:
```json
{
  "complaint_id": "generated_uuid",
  "priority_score": 0.668,
  "urgency_level": "High",
  "routing_recommendation": "Priority Maintenance Queue",
  "sla_hours": 2,
  "requires_immediate_action": false,
  "cluster_id": 7,
  "similar_complaints_count": 15,
  "decision_breakdown": {
    "model_urgency": 0.74,
    "cluster_density": 0.72,
    "temporal_boost": 0.30,
    "historical_calibration": 0.75,
    "semantic_similarity": 0.35
  },
  "explanation": "High priority due to model prediction (0.74), systemic issue detected (cluster #7 with 15 complaints in 3 days, density 0.72), peak hour filing (0.30), and historical pattern of delayed resolution (0.75)"
}
```

### Endpoint: POST `/batch-predict`

**Request**:
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      {"text": "AC broken", "timestamp": "2026-02-17T08:00:00Z"},
      {"text": "WiFi slow", "timestamp": "2026-02-17T09:00:00Z"}
    ]
  }'
```

**See [api/routes.py](api/routes.py) for complete API specification.**

---

## How It Works: Key Algorithms

### 1. CNN-BiLSTM Urgency Model

**Input**: Token sequence `[t₁, t₂, ..., tₙ]`  
**Process**:
1. Embed tokens using FastText: `E = [e₁, e₂, ..., eₙ]` (each `eᵢ ∈ ℝ³⁰⁰`)
2. Apply 1D convolution (kernel size 5) to extract local patterns
3. Feed into Bidirectional LSTM to capture context
4. Dense head outputs `[P(Low), P(Medium), P(High)]`

**Training**: Categorical cross-entropy loss, Adam optimizer, early stopping on validation loss

### 2. HDBSCAN Clustering for Systemic Issue Detection

**Input**: Sentence embeddings from vector database  
**Process**:
1. Run HDBSCAN (min_cluster_size=5) to auto-identify clusters
2. Calculate **spatial density**: `1 / avg_intra_cluster_distance`
3. Calculate **temporal density**: `complaints_per_hour` within cluster
4. Combined density score: `0.6 × spatial + 0.4 × temporal`

**Output**: Cluster ID, density score, outlier flag

**Insight**: High-density clusters indicate recurring/systemic issues → boost priority

### 3. Temporal Frequency Burst Detection

**Input**: Complaint timestamp + historical complaint rate  
**Process**:
1. Count complaints in last 24 hours: `C_24h`
2. Retrieve historical average: `C_avg`
3. If `C_24h > 2 × C_avg`: Burst detected → `+0.20` boost
4. If `C_24h > 1.5 × C_avg`: Elevated frequency → `+0.10` boost

**Example**: Normal = 5/day, Today = 15/day → 3x spike → Burst boost applied

### 4. Historical Resolution Feedback Loop

**Input**: Past complaints similar to current complaint (semantic search)  
**Process**:
1. Retrieve resolution times of top-10 similar cases
2. Calculate `avg_resolution_time`
3. Compare to SLA target for predicted urgency level
4. If `avg_resolution_time > 1.5 × SLA`: Upward calibration (`H_history = 0.80`)
5. If `avg_resolution_time < 0.8 × SLA`: Downward calibration (`H_history = 0.30`)

**Result**: System self-corrects without retraining

---

## Contributing

We welcome contributions! Areas of interest:

1. **Model Improvements**: Experiment with Transformer encoders (BERT, RoBERTa) for urgency classification
2. **Clustering Algorithms**: Compare HDBSCAN with OPTICS, DBSCAN variants
3. **Explainability**: Integrate attention visualization for model predictions
4. **Testing**: Expand unit test coverage (currently ~60%)
5. **Documentation**: Add tutorials, use-case examples

**Steps to Contribute**:
1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature X"`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

**Code Style**: Follow PEP 8, use type hints, docstrings for all functions.

---

## Testing

**Run all tests**:
```bash
bash run_tests.sh
```

**Run specific test suite**:
```bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_inference.py -v
```

**Coverage report**:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Acknowledgments

- **FastText**: Facebook AI Research for pretrained word embeddings
- **Sentence-BERT**: Nils Reimers for semantic sentence embeddings
- **HDBSCAN**: Leland McInnes for robust clustering algorithm
- **TensorFlow/Keras**: Google for deep learning framework
- **spaCy**: Explosion AI for NLP pipeline

## Project Status

🚀 **Status**: Active Development  
📅 **Last Updated**: February 2026  
🎯 **Current Version**: v1.2.0  
✅ **Production Ready**: Yes (with appropriate testing for your use case)

---

**Built with ❤️ for smarter hostel management**

```bash
python api/main.py
```

or

```bash
fastapi dev --app api.main:app
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Usage Examples

### Python API

```python
from src.inference import GrievancePredictor
from src.decision_engine import UrgencyDecisionEngine
from datetime import datetime

# Initialize predictor
predictor = GrievancePredictor(
    model_path="outputs/models/best_model.h5"
)

# Make prediction
result = predictor.predict(
    complaint_text="The water heater is broken!!!",
    timestamp=datetime.now(),
    hostel_id="H-101"
)

print(f"Urgency: {result['urgency_level']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### REST API

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
       "complaint_text": "The water heater is broken!!!",
       "timestamp": "2026-02-17T18:30:00",
       "hostel_id": "H-101"
     }'
```

## Model Architecture

### Text Processing Pipeline
1. **Cleaning**: Caps detection, emoji conversion, punctuation analysis
2. **Tokenization**: Lemmatization with custom stop word handling
3. **Embedding**: FastText word vectors (300-dim)

### Neural Network
- **Input**: Tokenized sequences (max length 100)
- **Embedding Layer**: Pretrained FastText weights
- **CNN Layer**: 128 filters, kernel size 5
- **BiLSTM Layers**: 64 units with dropout
- **Dense Layers**: 128 → 64 → 3 (urgency classes)

### Decision Engine
Rule-based adjustments based on:
- Temporal context (night/peak hours/weekend)
- Critical keywords (emergency, urgent, etc.)
- Complaint history (repeated issues)

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocessing.py

# With coverage
python -m pytest --cov=src tests/
```

## Documentation

See the `docs/` directory for detailed documentation:
- [architecture.md](docs/architecture.md) - System architecture
- [data_flow.md](docs/data_flow.md) - Data processing pipeline
- [decision_logic.md](docs/decision_logic.md) - Decision engine rules


## Acknowledgments

- FastText pretrained embeddings
- spaCy for NLP processing
- TensorFlow/Keras for deep learning
