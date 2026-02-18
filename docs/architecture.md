# System Architecture

## Overview

The **Hostel Grievance Urgency Classification and Routing System** is a multi-stage intelligent decision pipeline that combines deep learning, semantic vector search, temporal pattern analysis, and density-based clustering to automatically prioritize and route large volumes of hostel complaints. Unlike traditional text classifiers, this system operates as a **hybrid architecture** that jointly considers semantic similarity, inferred urgency, temporal context, historical resolution patterns, and cluster density to produce robust, context-aware priority scores.

## High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                   │
│              Raw Complaint Text + Metadata + Timestamp                │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  • Text Cleaning (emotion preservation)                      │    │
│  │  • Tokenization (urgency-aware stop words)                   │    │
│  │  • Temporal Encoding (JSON-based time features)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌────────────────────┐      ┌────────────────────────┐
│  BRANCH A:         │      │  BRANCH B:             │
│  URGENCY MODEL     │      │  SEMANTIC EMBEDDINGS   │
│                    │      │                        │
│  ┌──────────────┐ │      │  ┌──────────────────┐  │
│  │ Embedding    │ │      │  │ Transformer-     │  │
│  │ Layer        │ │      │  │ based Encoder    │  │
│  │ (FastText)   │ │      │  │ (sentence-level) │  │
│  └──────┬───────┘ │      │  └────────┬─────────┘  │
│         ▼         │      │           ▼            │
│  ┌──────────────┐ │      │  ┌──────────────────┐  │
│  │ CNN Layer    │ │      │  │ Vector Database  │  │
│  │ (local       │ │      │  │ Storage          │  │
│  │ patterns)    │ │      │  └────────┬─────────┘  │
│  └──────┬───────┘ │      │           ▼            │
│         ▼         │      │  ┌──────────────────┐  │
│  ┌──────────────┐ │      │  │ HDBSCAN          │  │
│  │ BiLSTM       │ │      │  │ Clustering       │  │
│  │ (contextual  │ │      │  │                  │  │
│  │ dependencies)│ │      │  │ • Cluster ID     │  │
│  └──────┬───────┘ │      │  │ • Density Score  │  │
│         ▼         │      │  │ • Core/Outlier   │  │
│  ┌──────────────┐ │      │  └────────┬─────────┘  │
│  │ Dense Head   │ │      │           │            │
│  │ (urgency     │ │      │           │            │
│  │ score)       │ │      └───────────┼────────────┘
│  └──────┬───────┘ │                  │
└─────────┼─────────┘                  │
          │                            │
          └────────────┬───────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    DECISION ENGINE (FUSION LAYER)                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Inputs:                                                      │    │
│  │  • Model Urgency Score (from CNN-BiLSTM)                     │    │
│  │  • Cluster Density Score (from HDBSCAN)                      │    │
│  │  • Temporal Frequency Pattern (JSON temporal encoding)       │    │
│  │  • Historical Resolution Feedback (avg resolution time)      │    │
│  │  • Semantic Similarity to High-Priority Cases (vector DB)    │    │
│  │                                                               │    │
│  │  Weighted Fusion Formula:                                     │    │
│  │  Priority = α·U_model + β·D_cluster + γ·T_temporal +         │    │
│  │             δ·H_history + ε·S_semantic                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                  │
│  • Final Priority Score (0.0 - 1.0)                                  │
│  • Urgency Level (Low / Medium / High / Critical)                    │
│  • Routing Recommendation (Department/Team)                          │
│  • Cluster Assignment (similar complaints)                           │
│  • Explanation Vector (contributory factors)                         │
└──────────────────────────────────────────────────────────────────────┘
```

## Core Components: Detailed Architecture

### 1. Preprocessing Pipeline

**Purpose**: Transform raw complaints into clean, structured representations while preserving urgency signals and extracting temporal context.

#### 1.1 Text Cleaning (`clean_text.py`)

**Key Operations**:
- **Emotional Signal Preservation**: Detects ALL CAPS, excessive punctuation (!!!, ???), emoji patterns
- **Normalization**: Lowercasing, contraction expansion, whitespace cleanup
- **Urgency Tagging**: Adds metadata tags like `[CAPS_ON]`, `[Multiple Exclamations]`, `[angry_face]`

**Design Rationale**: Traditional text cleaning removes emotional signals that are critical for urgency detection. Our approach preserves these while normalizing the rest.

#### 1.2 Tokenization (`tokenizer.py`)

**Features**:
- **Urgency-Aware Stop Words**: Preserves negations and urgency indicators (not, no, never, now, immediately)
- **Lemmatization**: Using spaCy for context-aware stemming
- **Special Token Handling**: Maintains tagged signals from cleaning phase

**Output**: Token sequences optimized for both deep learning and semantic search.

#### 1.3 Temporal Encoding (`temporal_encoder.py`)

**Extracted Features** (JSON format):
```json
{
  "hour": 0-23,
  "day_of_week": 0-6,
  "is_peak_hour": boolean,
  "is_weekend": boolean,
  "is_night": boolean,
  "month": 1-12,
  "day_of_month": 1-31,
  "complaint_frequency_last_24h": int,
  "complaint_frequency_last_week": int
}
```

**Novel Aspect**: Temporal patterns are encoded as structured JSON and stored separately, enabling temporal frequency analysis to detect complaint bursts.

---

### 2. Parallel Processing Branches

The system employs **parallel processing** with two distinct computational branches that run concurrently:

#### **Branch A: Urgency Inference Model (CNN-BiLSTM)**

This branch predicts urgency scores using a hybrid deep learning architecture.

##### 2.1 Embedding Layer

**Implementation**: Pretrained FastText embeddings (300-dimensional)

**Advantages**:
- Subword information handles out-of-vocabulary (OOV) words
- Domain-agnostic pretraining on large corpora
- Frozen weights prevent overfitting on small datasets

**Mathematical Representation**:
```
Input: token sequence T = [t₁, t₂, ..., tₙ]
Output: E = [e₁, e₂, ..., eₙ] where eᵢ ∈ ℝ³⁰⁰
```

##### 2.2 CNN Layer (Local Pattern Extraction)

**Purpose**: Extract **local n-gram patterns** that indicate urgency (e.g., "broken water heater", "immediate help needed")

**Architecture**:
- **Filters**: 128 convolutional filters
- **Kernel Size**: 5 (captures 5-gram patterns)
- **Activation**: ReLU
- **Padding**: Same (preserves sequence length)

**How CNN Extracts Local Patterns**:
Convolutional filters slide over the embedded sequence, detecting specific local patterns:
```
Filter 1 might activate for: ["urgent", "need", "help", "now", "please"]
Filter 2 might activate for: ["broken", "water", "heater"]
Filter 3 might activate for: ["cannot", "sleep", "noise"]
```

**Output**: Feature map capturing local urgency indicators across the sequence.

##### 2.3 BiLSTM Layer (Contextual Dependencies)

**Purpose**: Capture **long-range contextual dependencies** and understand how urgency evolves across the complaint narrative.

**Architecture**:
- **Bidirectional LSTM** with 64 units per direction
- **Two stacked layers** for hierarchical feature learning
- **Dropout**: 0.5 for regularization

**How BiLSTM Captures Context**:
- **Forward LSTM**: Reads complaint left-to-right, building context of what came before
- **Backward LSTM**: Reads right-to-left, incorporating future context
- **Combined Representation**: Concatenates both directions to understand full context

**Example**:
```
Text: "The AC stopped working and it's unbearably hot"
Context captured: "stopped working" + "unbearably hot" → high urgency
(BiLSTM understands the causal relationship between AC failure and discomfort)
```

**Mathematical Representation**:
```
h⃗ₜ = LSTM_forward(hₜ₋₁, eₜ)
h⃖ₜ = LSTM_backward(hₜ₊₁, eₜ)
hₜ = [h⃗ₜ; h⃖ₜ]  (concatenation)
```

##### 2.4 Dense Head (Urgency Prediction)

**Architecture**:
- Dense layer 1: 128 units + ReLU + Dropout
- Dense layer 2: 64 units + ReLU + Dropout
- Output layer: 3 units (Low, Medium, High) + Softmax

**Output**: Probability distribution over urgency classes
```
P(urgency) = [P(Low), P(Medium), P(High)]
Urgency Score U_model = weighted_average(P, weights=[0.0, 0.5, 1.0])
```

---

#### **Branch B: Semantic Embeddings & Clustering**

This branch creates semantic representations for similarity search and cluster density analysis.

##### 2.5 Transformer-Based Semantic Embedder

**Purpose**: Generate **sentence-level semantic embeddings** that capture deep contextual meaning.

**Implementation Options**:
- Sentence-BERT (SBERT)
- Universal Sentence Encoder (USE)
- MiniLM variants

**Process**:
```
Input: Preprocessed text
Output: Dense vector v ∈ ℝᵈ (d = 384 or 768 dimensions)
```

**Difference from FastText**:
- FastText: Averages word-level embeddings (bag-of-words with subword info)
- Transformer: Contextual sentence-level representation (semantically aware)

##### 2.6 Vector Database Storage

**Purpose**: Store embeddings for fast similarity search and clustering.

**Implementation**: Vector store with cosine similarity search

**Operations**:
- `add(vector, id, metadata)`: Store new complaint embedding
- `search(query_vector, top_k)`: Retrieve k most similar complaints
- `batch_retrieve(ids)`: Get embeddings for clustering

**Data Structure**:
```python
{
  "id": "complaint_12345",
  "vector": [0.23, -0.45, 0.67, ...],  # 384-dim
  "metadata": {
    "timestamp": "2026-02-17T14:30:00",
    "urgency_level": "High",
    "resolution_time": 2.5  # hours
  }
}
```

##### 2.7 HDBSCAN Clustering

**Purpose**: Group semantically similar complaints and compute **cluster density scores**.

**Algorithm**: HDBSCAN (Hierarchical Density-Based Spatial Clustering)

**Advantages Over K-Means**:
- No need to predefine number of clusters
- Identifies noise/outliers (single-issue complaints)
- Produces hierarchical cluster structure
- Density-based (adapts to cluster shape)

**Process**:
```
Input: Embedding vectors from vector database
Output: 
  - Cluster assignments: {complaint_id: cluster_id}
  - Cluster densities: {cluster_id: density_score}
  - Outlier flags: {complaint_id: is_outlier}
```

**How Cluster Density Affects Priority**:
- **High-density clusters**: Many similar complaints → systemic issue → higher priority
- **Outliers**: Single unique complaint → normal priority
- **Growing clusters**: Rapidly increasing density → emerging issue → urgent attention

**Density Score Calculation**:
```
D_cluster = (n_complaints_in_cluster / time_window) × (1 / avg_inter_complaint_distance)
```

---

### 3. Decision Engine (Fusion Layer)

**Purpose**: Combine signals from both branches plus temporal/historical data into a single priority score.

#### 3.1 Input Signals

| Signal | Source | Range | Meaning |
|--------|--------|-------|---------|
| U_model | CNN-BiLSTM output | [0, 1] | Model-predicted urgency |
| D_cluster | HDBSCAN density | [0, ∞) | Cluster density score |
| T_temporal | Temporal encoding | [0, 1] | Temporal urgency boost |
| H_history | Resolution database | [0, 1] | Historical urgency calibration |
| S_semantic | Vector similarity | [0, 1] | Similarity to past high-priority cases |

#### 3.2 Fusion Formula

**Weighted Aggregation**:
```
Priority = α·U_model + β·D_cluster + γ·T_temporal + δ·H_history + ε·S_semantic

where:
  α + β + γ + δ + ε = 1.0 (normalized weights)
  
Default weights:
  α = 0.40  (model urgency - primary signal)
  β = 0.25  (cluster density - systemic issues)
  γ = 0.15  (temporal context)
  δ = 0.10  (historical feedback)
  ε = 0.10  (semantic similarity to critical cases)
```

#### 3.3 Adaptive Calibration

**Historical Feedback Loop**:
- Track actual resolution times for each urgency level
- If High-urgency complaints consistently resolve slowly → decrease threshold
- If Low-urgency complaints escalate → increase sensitivity

**Temporal Frequency Adjustment**:
- Detect complaint bursts (spike in D_cluster or temporal frequency)
- Dynamically increase β (cluster weight) during burst periods
- Return to normal weights after burst subsides

---

### 4. Routing Decision Logic

**Post-Priority Routing**:

```
IF Priority > 0.85:
    → Route to: Emergency Response Team
    → SLA: 30 minutes
ELSE IF Priority > 0.65:
    → Route to: Priority Maintenance Queue
    → SLA: 2 hours
ELSE IF Cluster_ID in systemic_issues:
    → Route to: Facilities Management (batch resolution)
    → SLA: 24 hours
ELSE:
    → Route to: Standard Queue
    → SLA: 48 hours
```

---

## Parallel Processing Design

**Computational Efficiency**:

```
Branch A (CNN-BiLSTM) and Branch B (Embeddings + Clustering) run in PARALLEL:

Time Complexity:
  Sequential: T(Branch A) + T(Branch B) = 150ms + 80ms = 230ms
  Parallel: max(T(Branch A), T(Branch B)) = 150ms

Speedup: 1.53x
```

**Implementation**:
- Asynchronous processing using concurrent futures
- Branch A runs on GPU (if available)
- Branch B runs on CPU (vector operations)
- Decision engine waits for both branches to complete

---

## Step-by-Step Flow: Complaint Input → Priority Output

**Complete Pipeline Execution**:

```
1. INPUT
   ↓ Complaint: "Water heater broken since morning, no hot water!!!"
   ↓ Timestamp: 2026-02-17 08:30:00

2. PREPROCESSING
   ↓ Cleaned: "[Multiple Exclamations] water heater broken since morning no hot water"
   ↓ Tokens: ["water", "heater", "broken", "morning", "hot", "water"]
   ↓ Temporal: {hour: 8, is_peak_hour: true, is_weekend: false}

3. BRANCH A (CNN-BiLSTM)
   ↓ Embeddings: [[0.12, -0.34, ...], [0.56, 0.23, ...], ...]
   ↓ CNN features: Detects "broken" + "heater" pattern
   ↓ BiLSTM context: Understands "no hot water" consequence
   ↓ U_model = 0.72 (High urgency)

4. BRANCH B (Semantic)
   ↓ Sentence embedding: [0.45, -0.12, 0.89, ...]
   ↓ Vector search: 15 similar complaints in last week
   ↓ HDBSCAN: Cluster ID = 7, Density = 0.68
   ↓ D_cluster = 0.68 (growing cluster → systemic issue)

5. DECISION ENGINE
   ↓ T_temporal = 0.50 (morning peak hour)
   ↓ H_history = 0.60 (water heater issues historically urgent)
   ↓ S_semantic = 0.75 (similar to past critical cases)
   ↓ 
   ↓ Priority = 0.40×0.72 + 0.25×0.68 + 0.15×0.50 + 0.10×0.60 + 0.10×0.75
   ↓          = 0.288 + 0.170 + 0.075 + 0.060 + 0.075
   ↓          = 0.668

6. OUTPUT
   ↓ Priority Score: 0.668
   ↓ Urgency Level: High
   ↓ Routing: Priority Maintenance Queue
   ↓ SLA: 2 hours
   ↓ Cluster: #7 (Water Heater Issues - 15 complaints this week)
   ↓ Explanation: "High urgency due to model prediction (0.72), systemic issue detected (cluster density 0.68), and peak hour filing"
```

---

## Novel Architectural Features (Patent-Worthy)

1. **Parallel Dual-Branch Processing**: Simultaneous urgency inference and semantic clustering
2. **Density-Aware Prioritization**: Using HDBSCAN cluster density to detect systemic issues
3. **Temporal Frequency Analysis**: JSON-based temporal encoding to detect complaint bursts
4. **Adaptive Historical Calibration**: Self-adjusting urgency thresholds based on resolution feedback
5. **Multi-Signal Fusion**: Weighted combination of 5 distinct signals for robust priority scoring
6. **Emotion-Preserving Preprocessing**: Maintains urgency signals during text normalization
7. **Hierarchical Decision Logic**: Model-based urgency refined by contextual rules

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | TensorFlow/Keras |
| Word Embeddings | FastText (300-dim) |
| Sentence Embeddings | Sentence-BERT / Universal Sentence Encoder |
| Clustering | HDBSCAN |
| Vector Database | Custom implementation (scalable to FAISS/Pinecone) |
| NLP Processing | spaCy, NLTK |
| API Framework | FastAPI |
| Task Orchestration | Python asyncio |

---

## Scalability Considerations

**Current Capacity**: ~10,000 complaints/day
**Scaling Path**:
- Vector DB → FAISS GPU for million-scale similarity search
- CNN-BiLSTM → TensorFlow Serving for batch inference
- Decision Engine → Redis cache for temporal frequency lookups
- Clustering → Incremental HDBSCAN for real-time cluster updates

**Latency Targets**:
- Preprocessing: <50ms
- Model Inference: <150ms
- Semantic Search: <30ms
- Clustering Update: <100ms (async)
- Total End-to-End: <250ms (P95)
    ▼
Conv1D (128 filters, kernel_size=5)
    │
    ▼
BiLSTM (64 units) × 2
    │
    ▼
Dense (128 → 64 → 3)
    │
    ▼
Softmax (Low, Medium, High)
```

**Why CNN-BiLSTM?**
- CNN captures local patterns (keywords, phrases)
- BiLSTM captures context and word order
- Proven effective for text classification

**Alternative Architectures** (in `src/models/`):
- Pure CNN: Faster, good for keyword-heavy tasks
- Pure BiLSTM: Better for context-dependent text

### 4. Decision Engine

**Purpose**: Refine ML predictions with domain-specific rules.

**Adjustments**:

| Factor | Boost | Condition |
|--------|-------|-----------|
| Night-time (12am-6am) | +0.15 | is_night=True |
| Peak hours (6pm-12am) | +0.10 | is_peak_hour=True |
| Weekend | +0.05 | is_weekend=True |
| Critical keywords | +0.1-0.3 | "emergency", "urgent", etc. |
| Repeated complaint | +0.20 | History count > 0 |

**Logic**:
```python
adjusted_confidence = base_confidence + sum(boosts)
if adjusted_confidence >= threshold:
    escalate_urgency()
```

### 5. API Layer

**FastAPI Application**:
- `/api/v1/predict` - Single complaint prediction
- `/api/v1/predict/batch` - Batch predictions
- `/health` - Health check
- `/stats` - Model statistics

**Request/Response Flow**:
1. Validate input (Pydantic schemas)
2. Preprocess text
3. Run model inference
4. Apply decision engine
5. Return structured response

## Data Flow

### Training Flow

```
Raw Data → Clean → Tokenize → Embed → Train → Validate → Save Model
```

### Inference Flow

```
Request → Preprocess → Embed → Predict → Decide → Response
```

## Scalability Considerations

### Current Architecture (v0.1)
- Single-instance API
- In-memory model loading
- Sequential batch processing

### Future Enhancements
1. **Horizontal Scaling**
   - Load balancer + multiple API instances
   - Shared model cache (Redis)
   
2. **Model Serving**
   - TensorFlow Serving
   - TorchServe integration
   
3. **Async Processing**
   - Celery for batch jobs
   - Message queue (RabbitMQ/Kafka)
   
4. **Database Integration**
   - PostgreSQL for complaint history
   - Vector DB for similarity search

## Performance

### Latency Targets
- Single prediction: < 100ms
- Batch (100 items): < 5s

### Throughput
- Current: ~50 requests/second
- Target: 500+ requests/second (with scaling)

## Security

- Input validation with Pydantic
- Rate limiting (future)
- API key authentication (future)
- Encrypted model storage (future)

## Monitoring

### Metrics to Track
- Prediction latency (p50, p95, p99)
- Model accuracy drift
- API error rates
- Resource utilization (CPU, memory)

### Tools
- TensorBoard for training
- Prometheus + Grafana for production
- Logging with structured output

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+ |
| ML Framework | TensorFlow/Keras |
| API | FastAPI |
| NLP | spaCy, FastText |
| Data | Pandas, NumPy |
| Testing | pytest |
| Deployment | Docker (future) |

## Configuration Management

Centralized in `src/config.py`:
- Model hyperparameters
- Training configuration
- File paths
- API settings
- Logging configuration

## Error Handling

- Input validation errors → 400 Bad Request
- Model not found → 503 Service Unavailable
- Prediction errors → 500 Internal Server Error
- Structured error responses with details
