# Data Flow Documentation

## Overview

This document provides a comprehensive view of the **data lifecycle** in the Hostel Grievance Urgency Classification System, from raw complaint ingestion to final priority output. The system maintains a structured data organization that separates concerns, enables reproducibility, and supports both training and inference pipelines.

## Data Directory Structure

```
data/
├── raw/                          # Immutable raw data
│   ├── complaints/               # Original complaint texts
│   │   ├── batch_001.txt
│   │   ├── batch_002.json
│   │   └── streaming/            # Real-time complaint feed
│   └── metadata/                 # Associated metadata files
│       ├── complaint_meta.csv    # Timestamps, hostel_id, etc.
│       └── resolution_history.db # Historical resolution data
│
├── processed/                    # Transformed data artifacts
│   ├── tokenized/                # Cleaned and tokenized text
│   │   ├── train_tokens.pkl
│   │   ├── val_tokens.pkl
│   │   └── vocabulary.json
│   ├── embeddings/               # Vector representations
│   │   ├── fasttext_sequences.npy      # For CNN-BiLSTM
│   │   ├── sentence_embeddings.npy     # For semantic search
│   │   ├── embedding_matrix.npy        # Pretrained weights
│   │   └── vector_store.pkl            # Vector database
│   └── temporal_json/            # Temporal feature encodings
│       ├── temporal_features.json
│       └── frequency_patterns.json
│
└── external/                     # External datasets/models
    ├── cc.en.300.bin             # FastText pretrained model
    ├── sbert_model/              # Sentence-BERT checkpoint
    └── stopwords_custom.txt      # Custom stop words list

outputs/
├── models/                       # Trained model checkpoints
│   ├── cnn_bilstm_best.h5
│   ├── tokenizer.pkl
│   └── scaler.pkl
├── logs/                         # Training logs & metrics
│   ├── training_history.csv
│   └── tensorboard/
└── reports/                      # Inference outputs
    ├── predictions.json
    ├── cluster_analysis.html
    └── urgency_distribution.png
```

---

## Pipeline Stages: Complete Data Lifecycle

### Stage 1: Data Ingestion

**Input Sources**:
- Batch uploads: Text files, CSV, JSON
- Real-time stream: API endpoint submissions
- Historical archive: Past complaints with resolved labels

**Raw Data Format**:
```json
{
  "complaint_id": "C_20260217_001",
  "text": "The water heater in room 204 has been broken since yesterday morning!!!",
  "timestamp": "2026-02-17T08:30:45Z",
  "metadata": {
    "hostel_id": "H-101",
    "room_number": "204",
    "student_id": "ANON_54321",
    "complaint_type": "maintenance",
    "previous_complaints": 2,
    "contact_info": "encrypted"
  }
}
```

**Storage**: `data/raw/complaints/`

**Metadata Tracking** (`data/raw/metadata/`):
- `complaint_meta.csv`: Master record of all complaints
- `resolution_history.db`: SQLite database with resolution outcomes
  - Fields: complaint_id, assigned_to, resolution_time_hours, final_urgency, satisfaction_score

---

### Stage 2: Preprocessing Pipeline

**Transformation**: Raw text → Clean tokens + Temporal features

#### 2.1 Text Cleaning (`clean_text.py`)

**Input**: Raw complaint text
**Output**: Cleaned text with preserved urgency signals
**Storage**: In-memory (passed to tokenizer)

**Transformations**:

| Input | Output | Preserved Signal |
|-------|--------|------------------|
| "HELP ME NOW!!!" | "[CAPS_ON] help me now [Multiple Exclamations]" | CAPS + emotion |
| "Water heater broken 😡" | "water heater broken [angry_face]" | Emoji sentiment |
| "Can't sleep, noise!!!" | "cannot sleep noise [Multiple Exclamations]" | Negation + emotion |
| "When??? No response" | "when [Impatient Questioning] no response" | Frustration + negation |

**Processing Pipeline**:
```python
1. detect_caps_emphasis(text)           # Tag ALL CAPS sections
2. convert_emojis_to_text(text)         # Map emojis to descriptive text
3. detect_punctuation_barrage(text)     # Tag excessive punctuation
4. expand_contractions(text)            # can't → cannot
5. clean_special_chars(text)            # Remove noise but keep urgency tags
6. normalize_whitespace(text)           
7. lowercase_except_tags(text)          # Lowercase but preserve tags
```

**Example**:
```
Raw: "URGENT!!! The AC in room 305 is broken and it's 🔥 hot can't sleep"
     ↓
Cleaned: "[CAPS_ON] urgent [Multiple Exclamations] the ac in room 305 is broken and it is [fire] hot cannot sleep"
```

#### 2.2 Tokenization (`tokenizer.py`)

**Input**: Cleaned text
**Output**: Token sequences with urgency-aware filtering
**Storage**: `data/processed/tokenized/`

**Process**:
```python
1. Load spaCy model (en_core_web_sm)
2. Apply custom stop word list (preserve negations: not, no, never, etc.)
3. Lemmatize tokens (running → run, heaters → heater)
4. Filter non-essential words (keep urgency indicators)
5. Preserve special tags ([CAPS_ON], [Multiple Exclamations])
```

**Example**:
```
Input:  "[CAPS_ON] urgent [Multiple Exclamations] the ac in room 305 is broken and it is [fire] hot cannot sleep"
        ↓
Tokens: ["[CAPS_ON]", "urgent", "[Multiple Exclamations]", "ac", "room", "305", "broken", "[fire]", "hot", "cannot", "sleep"]
```

**Vocabulary Construction**:
- Build word → index mapping from training data
- Save as `vocabulary.json`
```json
{
  "urgent": 145,
  "broken": 67,
  "cannot": 89,
  "[CAPS_ON]": 5,
  "[Multiple Exclamations]": 6,
  ...
}
```

**Output Files**:
- `train_tokens.pkl`: Tokenized training complaints
- `val_tokens.pkl`: Tokenized validation complaints
- `vocabulary.json`: Word-to-index mapping

#### 2.3 Temporal Encoding (`temporal_encoder.py`)

**Input**: Complaint timestamp + metadata
**Output**: Structured JSON temporal features
**Storage**: `data/processed/temporal_json/`

**Extracted Features**:
```json
{
  "complaint_id": "C_20260217_001",
  "temporal": {
    "hour": 8,
    "day_of_week": 1,          // Monday = 0, Sunday = 6
    "is_peak_hour": true,      // 6pm-12am or 7am-10am
    "is_weekend": false,
    "is_night": false,         // 12am-6am
    "month": 2,
    "day_of_month": 17,
    "week_of_year": 7
  },
  "frequency": {
    "complaints_last_24h": 3,       // From same hostel
    "complaints_last_week": 15,
    "complaint_rate_trend": "increasing"  // stable, increasing, decreasing
  },
  "context": {
    "hostel_id": "H-101",
    "complaint_type": "maintenance",
    "previous_count": 2          // From same student
  }
}
```

**Temporal Frequency Analysis**:
The system maintains a **rolling temporal index** to detect complaint bursts:
```python
# Pseudocode for frequency calculation
def calculate_frequency(hostel_id, timestamp):
    recent_complaints = query_db(
        "SELECT COUNT(*) FROM complaints 
         WHERE hostel_id = ? AND timestamp > ?",
        hostel_id, timestamp - 24_hours
    )
    return recent_complaints
```

**Output File**: `temporal_features.json` (one JSON object per complaint)

---

### Stage 3: Dual-Track Embedding Generation

This stage splits into **two parallel tracks**:

#### Track A: Word-Level Embeddings (for CNN-BiLSTM)

**Input**: Token sequences from Stage 2.2
**Output**: Sequential embedding matrices
**Storage**: `data/processed/embeddings/fasttext_sequences.npy`

**Process**:
```python
1. Load pretrained FastText model (data/external/cc.en.300.bin)
2. Create embedding matrix for vocabulary:
   embedding_matrix[word_index[word]] = fasttext_model[word]
3. Convert token sequences to padded index sequences:
   ["urgent", "broken", "ac"] → [145, 67, 892] → pad to max_length
4. Save embedding matrix and sequences
```

**File Structure**:
```python
# fasttext_sequences.npy
X_train.shape = (N_train, max_length)  # Integer indices
X_val.shape = (N_val, max_length)

# embedding_matrix.npy
shape = (vocab_size, 300)  # FastText 300-dim vectors
```

#### Track B: Sentence-Level Embeddings (for Semantic Search)

**Input**: Cleaned text (not token sequences)
**Output**: Dense sentence vectors
**Storage**: `data/processed/embeddings/sentence_embeddings.npy`

**Process**:
```python
1. Load Sentence-BERT or Universal Sentence Encoder
2. Encode entire complaint text as single vector:
   text → transformer → pooling → dense_vector (384 or 768 dim)
3. Store in vector database for similarity search
```

**Example**:
```python
Text: "The AC in room 305 is broken and it's hot, cannot sleep"
      ↓
Sentence Embedding: [0.234, -0.456, 0.789, ..., 0.123]  # 384-dim vector
```

**Vector Store Structure** (`data/processed/embeddings/vector_store.pkl`):
```python
{
  "vectors": np.array([[...], [...], ...]),  # (N, 384) matrix
  "ids": ["C_001", "C_002", ...],
  "metadata": [
    {"urgency": "High", "cluster_id": 7, "timestamp": "..."},
    ...
  ]
}
```

---

### Stage 4: Training Pipeline (Model Development)

**Purpose**: Train CNN-BiLSTM on labeled urgency data

**Input Data**:
- `data/processed/tokenized/train_tokens.pkl`
- `data/processed/embeddings/embedding_matrix.npy`
- Labels: Ground truth urgency levels (Low/Medium/High)

**Training Script**: `scripts/train_model.py`

**Process**:
```python
1. Load tokenized sequences and labels
2. Load pretrained embedding matrix
3. Initialize CNN-BiLSTM model
4. Compile with categorical crossentropy loss
5. Train with early stopping (validation loss)
6. Save best model to outputs/models/cnn_bilstm_best.h5
```

**Output Files**:
```
outputs/
├── models/
│   ├── cnn_bilstm_best.h5        # Trained model weights
│   ├── tokenizer.pkl              # Fitted tokenizer
│   └── label_encoder.pkl          # Urgency label mapping
└── logs/
    ├── training_history.csv       # Epoch-wise metrics
    └── tensorboard/               # Training visualizations
```

**Training Logs** (`training_history.csv`):
```csv
epoch,loss,accuracy,val_loss,val_accuracy
1,1.034,0.456,0.987,0.489
2,0.876,0.567,0.854,0.601
...
```

---

### Stage 5: Embedding Database Construction

**Purpose**: Build searchable vector database for semantic similarity

**Input**: Sentence embeddings from Stage 3 Track B
**Script**: `scripts/build_embeddings.py`

**Process**:
```python
1. Load all sentence embeddings
2. Initialize vector store with dimensionality
3. Add each embedding with metadata:
   vector_store.add(
       vector=embedding,
       id=complaint_id,
       metadata={
           "timestamp": timestamp,
           "urgency_level": predicted_urgency,
           "cluster_id": None  # Populated later
       }
   )
4. Save vector store to disk
```

**Clustering with HDBSCAN**:
```python
1. Extract all vectors from vector store
2. Run HDBSCAN clustering:
   clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
   cluster_labels = clusterer.fit_predict(vectors)
3. Calculate cluster density scores:
   for cluster_id in unique_labels:
       density = calculate_density(cluster_id, vectors, timestamps)
4. Update vector store metadata with cluster assignments
5. Save cluster analysis to outputs/reports/cluster_analysis.html
```

**Density Calculation**:
```python
def calculate_density(cluster_id, vectors, timestamps):
    cluster_vectors = vectors[labels == cluster_id]
    cluster_times = timestamps[labels == cluster_id]
    
    # Spatial density (avg inter-point distance)
    distances = pairwise_distances(cluster_vectors)
    avg_distance = np.mean(distances)
    spatial_density = 1 / (avg_distance + 1e-6)
    
    # Temporal density (complaints per hour)
    time_span = (max(cluster_times) - min(cluster_times)).total_seconds() / 3600
    temporal_density = len(cluster_vectors) / max(time_span, 1.0)
    
    # Combined density
    return 0.6 * spatial_density + 0.4 * temporal_density
```

**Output**:
- Updated `vector_store.pkl` with cluster metadata
- `cluster_analysis.html`: Visualization of clusters
- `cluster_density_scores.json`: Density scores by cluster

---

### Stage 6: Inference Pipeline (Production)

**Purpose**: Process new complaints in real-time

**Input**: Single complaint (JSON from API or file)
**Script**: `scripts/run_inference.py`

**Complete Data Flow**:

```
┌─────────────────────────────────────────────────────────────────┐
│ NEW COMPLAINT                                                    │
│ "Water heater broken in room 204, no hot water since morning!!!"│
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │ PREPROCESSING          │
        │ • Clean text           │
        │ • Tokenize             │
        │ • Extract temporal     │
        └───────┬────────────────┘
                │
                ├──────────────────────┬───────────────────────┐
                ▼                      ▼                       ▼
    ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────┐
    │ BRANCH A:         │  │ BRANCH B:         │  │ TEMPORAL           │
    │ CNN-BiLSTM        │  │ Semantic Search   │  │ FEATURES           │
    │                   │  │                   │  │                    │
    │ 1. Convert to     │  │ 1. Generate       │  │ • Hour: 8          │
    │    token indices  │  │    sentence       │  │ • is_peak: True    │
    │ 2. Pad sequence   │  │    embedding      │  │ • is_night: False  │
    │ 3. Model predict  │  │ 2. Search vector  │  │ • weekend: False   │
    │                   │  │    database       │  │                    │
    │ Output:           │  │ 3. Get top-5      │  │ Frequency:         │
    │ U_model = 0.72    │  │    similar cases  │  │ • Last 24h: 3      │
    │ (High urgency)    │  │ 4. Run HDBSCAN    │  │ • Last week: 15    │
    │                   │  │                   │  │                    │
    │                   │  │ Output:           │  │ Output:            │
    │                   │  │ D_cluster = 0.68  │  │ T_temporal = 0.50  │
    │                   │  │ S_semantic = 0.75 │  │                    │
    └───────────────────┘  └───────────────────┘  └────────────────────┘
                │                      │                       │
                └──────────────────────┴───────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │ DECISION ENGINE          │
                        │                          │
                        │ Load historical data:    │
                        │ H_history = 0.60         │
                        │                          │
                        │ Fusion Formula:          │
                        │ P = 0.40×0.72 +          │
                        │     0.25×0.68 +          │
                        │     0.15×0.50 +          │
                        │     0.10×0.60 +          │
                        │     0.10×0.75            │
                        │   = 0.668                │
                        └──────────┬───────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────┐
                        │ OUTPUT                   │
                        │ • Priority: 0.668        │
                        │ • Level: High            │
                        │ • Route: Priority Queue  │
                        │ • Cluster: #7            │
                        │ • SLA: 2 hours           │
                        └──────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────┐
                        │ SAVE RESULTS             │
                        │ outputs/reports/         │
                        │ predictions.json         │
                        └──────────────────────────┘
```

**Output File** (`outputs/reports/predictions.json`):
```json
{
  "complaint_id": "C_20260217_001",
  "timestamp": "2026-02-17T08:30:45Z",
  "text": "Water heater broken in room 204, no hot water since morning!!!",
  "preprocessing": {
    "cleaned_text": "[Multiple Exclamations] water heater broken in room 204 no hot water since morning",
    "tokens": ["water", "heater", "broken", "room", "204", "hot", "water", "morning"]
  },
  "model_prediction": {
    "urgency_score": 0.72,
    "predicted_class": "High",
    "class_probabilities": {
      "Low": 0.08,
      "Medium": 0.20,
      "High": 0.72
    }
  },
  "semantic_analysis": {
    "cluster_id": 7,
    "cluster_density": 0.68,
    "similar_complaints": [
      {"id": "C_20260215_089", "similarity": 0.89},
      {"id": "C_20260216_034", "similarity": 0.85},
      ...
    ],
    "semantic_similarity_score": 0.75
  },
  "temporal_analysis": {
    "hour": 8,
    "is_peak_hour": true,
    "is_weekend": false,
    "is_night": false,
    "frequency_last_24h": 3,
    "temporal_urgency_score": 0.50
  },
  "historical_analysis": {
    "avg_resolution_time_similar": 2.3,
    "historical_urgency_calibration": 0.60
  },
  "final_decision": {
    "priority_score": 0.668,
    "urgency_level": "High",
    "routing_recommendation": "Priority Maintenance Queue",
    "sla_hours": 2,
    "requires_immediate_action": false,
    "explanation": "High priority due to: model prediction (0.72), systemic issue detected (cluster #7 with 15 complaints this week, density 0.68), peak hour filing (0.50)"
  }
}
```

---

## Data Reproducibility & Version Control

### Scripts for Reproducibility

**`scripts/train_model.py`**:
```bash
python scripts/train_model.py \
  --data data/processed/tokenized \
  --embeddings data/processed/embeddings/embedding_matrix.npy \
  --model-type cnn_bilstm \
  --epochs 50 \
  --batch-size 32 \
  --output outputs/models/
```

**`scripts/build_embeddings.py`**:
```bash
python scripts/build_embeddings.py \
  --input data/processed/tokenized/all_complaints.pkl \
  --model-path data/external/sbert_model \
  --output data/processed/embeddings/vector_store.pkl \
  --run-clustering
```

**`scripts/run_inference.py`**:
```bash
python scripts/run_inference.py \
  --model outputs/models/cnn_bilstm_best.h5 \
  --vector-store data/processed/embeddings/vector_store.pkl \
  --input data/raw/complaints/new_complaint.txt \
  --output outputs/reports/predictions.json
```

### Data Versioning

**Approach**: Git LFS for large files + metadata versioning

```
data/raw/                    # Git LFS tracked
data/processed/embeddings/   # Git LFS tracked
outputs/models/              # Git LFS tracked versioned snapshots

data/processed/tokenized/    # Git tracked (text-based)
outputs/reports/             # Git tracked (analysis outputs)
```

**Metadata Version Tracking**:
```json
{
  "version": "v1.2.0",
  "created": "2026-02-17",
  "data_sources": {
    "complaints": "data/raw/complaints/batch_001-050.txt",
    "fasttext_model": "cc.en.300.bin (2023-06-01)",
    "sbert_model": "all-MiniLM-L6-v2"
  },
  "preprocessing_config": {
    "max_length": 100,
    "min_token_length": 2,
    "stop_words_preserved": ["not", "no", "never", "now"]
  },
  "model_config": {
    "architecture": "cnn_bilstm",
    "training_samples": 15234,
    "validation_samples": 3809
  }
}
```

---

## Data Quality & Monitoring

**Automated Data Quality Checks**:
1. **Completeness**: Check for missing timestamps, metadata
2. **Consistency**: Validate complaint_id uniqueness
3. **Drift Detection**: Monitor token distribution shifts
4. **Clustering Health**: Track cluster count and density trends

**Monitoring Outputs** (`outputs/reports/`):
- `data_quality_report.html`: Daily data quality metrics
- `cluster_drift_analysis.png`: Cluster evolution over time
- `urgency_distribution.png`: Distribution of predicted urgency levels

---

## Summary: Data Artifacts by Stage

| Stage | Input | Output | Storage Location |
|-------|-------|--------|------------------|
| 1. Ingestion | Raw complaints | JSON/CSV files | `data/raw/complaints/` |
| 2. Preprocessing | Raw text | Cleaned tokens + temporal JSON | `data/processed/tokenized/`, `temporal_json/` |
| 3. Embedding (Track A) | Tokens | FastText sequences + matrix | `data/processed/embeddings/fasttext_*` |
| 3. Embedding (Track B) | Cleaned text | Sentence vectors + vector store | `data/processed/embeddings/sentence_*` |
| 4. Training | Tokenized data | Trained model weights | `outputs/models/` |
| 5. Clustering | Sentence vectors | Cluster assignments + densities | Updated `vector_store.pkl` |
| 6. Inference | New complaint | Priority prediction JSON | `outputs/reports/predictions.json` |

**Total Data Flow**: ~7 transformation stages, 12+ intermediate artifacts, fully reproducible via scripts.
          ↓
vectors = [
    [0.123, -0.456, ...],  # water
    [0.789, 0.234, ...],   # heater  
    [-0.345, 0.567, ...]   # broken
]
          ↓
Average pooling (for sentence embedding)
          ↓
sentence_vector = [0.189, 0.115, ...]
```

**Formats**:
- JSON: Individual complaint embeddings
- NumPy (.npy): Batch arrays for training
- Vector Store (.pkl): Similarity search index

### Stage 4: Sequence Preparation

**For Training/Inference**:

1. **Vocabulary Building** (training only):
   ```python
   tokenizer = Tokenizer(num_words=10000)
   tokenizer.fit_on_texts(all_tokens)
   word_index = tokenizer.word_index
   ```

2. **Text to Sequences**:
   ```python
   sequences = tokenizer.texts_to_sequences(tokens)
   # ["water", "heater", "broken"] → [234, 567, 890]
   ```

3. **Padding**:
   ```python
   padded = pad_sequences(sequences, maxlen=100)
   # [234, 567, 890] → [0, 0, ..., 234, 567, 890]
   #  ← 97 zeros →        ← actual sequence →
   ```

### Stage 5: Model Prediction

**Input**: Padded sequences (batch_size, 100)
**Output**: Class probabilities (batch_size, 3)

#### Forward Pass:

```
Sequence: [0, 0, ..., 234, 567, 890]
              ↓
Embedding Layer: (100, 300)
              ↓
Conv1D: (100, 128)
              ↓
BiLSTM: (100, 128)
              ↓
BiLSTM: (64,)
              ↓
Dense: (128) → (64) → (3)
              ↓
Softmax: [0.15, 0.35, 0.50]
         Low   Med   High
```

**Raw Prediction**:
```json
{
  "urgency_level": "High",
  "confidence": 0.50,
  "probabilities": {
    "Low": 0.15,
    "Medium": 0.35,
    "High": 0.50
  }
}
```

### Stage 6: Decision Engine Refinement

**Input**: Model prediction + temporal features
**Output**: Adjusted urgency with reasoning

#### Decision Logic:

```python
base_confidence = 0.50

# Temporal boost
if is_night:
    boost += 0.15  # → 0.65

# Keyword boost
if "emergency" in text:
    boost += 0.20  # → 0.85

# History boost
if previous_complaints > 0:
    boost += 0.20  # → 1.05 (capped at 1.0)

adjusted_confidence = min(1.0, base_confidence + boost)
```

**Final Output**:
```json
{
  "urgency_level": "High",
  "confidence": 0.85,
  "original_urgency": "High",
  "original_confidence": 0.50,
  "adjustments": [
    {
      "type": "temporal",
      "boost": 0.15,
      "reason": "Complaint filed during night-time"
    },
    {
      "type": "keyword",
      "boost": 0.20,
      "reason": "Critical keywords detected"
    }
  ],
  "total_boost": 0.35,
  "requires_immediate_action": true
}
```

### Stage 7: Output Generation

**Formats**:

1. **API Response (JSON)**:
   ```json
   {
     "success": true,
     "data": { ... },
     "processing_time_ms": 45.2
   }
   ```

2. **Batch Report (JSON file)**:
   ```json
   [
     {
       "id": 0,
       "text": "...",
       "urgency_level": "High",
       "confidence": 0.85
     },
     ...
   ]
   ```

3. **Logs** (`outputs/logs/`):
   ```
   2026-02-17 18:30:45 - INFO - Processed complaint #123
   2026-02-17 18:30:45 - INFO - Urgency: High (0.85)
   ```

## Data Formats

### Tokenized Data Format

```json
{
  "id": "complaint_001",
  "text": "original text",
  "cleaned_text": "cleaned text",
  "tokens": ["token1", "token2"],
  "metadata": {
    "timestamp": "2026-02-17T18:30:00",
    "hostel_id": "H-101"
  }
}
```

### Training Data Format

```python
# Features (X)
X = np.array([
    [0, 0, ..., 234, 567, 890],  # Complaint 1
    [0, 0, ..., 123, 456, 789],  # Complaint 2
    ...
])  # Shape: (num_samples, max_length)

# Labels (y) - one-hot encoded
y = np.array([
    [1, 0, 0],  # Low
    [0, 0, 1],  # High
    ...
])  # Shape: (num_samples, 3)
```

## Processing Statistics

### Typical Processing Times

| Stage | Single Item | Batch (100) |
|-------|-------------|-------------|
| Cleaning | 5ms | 200ms |
| Tokenization | 10ms | 500ms |
| Embedding | 2ms | 100ms |
| Model Prediction | 20ms | 800ms |
| Decision Engine | 1ms | 50ms |
| **Total** | **~40ms** | **~1.7s** |

### Data Reduction

```
Raw Complaint (avg):        150 characters
Cleaned Text:               120 characters (-20%)
Tokens:                     15 tokens (-88%)
Embedding:                  1 × 300 floats (1.2KB)
Padded Sequence:            100 integers (400 bytes)
```

## Data Validation

### Input Validation

- Text length: 10-1000 characters
- Timestamp: Valid datetime
- Hostel ID: Matches pattern
- Complaint type: From predefined list

### Output Validation

- Urgency level: Low | Medium | High
- Confidence: 0.0-1.0
- Probabilities sum to 1.0

## Error Handling

| Error Type | Action |
|------------|--------|
| Empty text | Return default "Low" urgency |
| Invalid timestamp | Use current time |
| Missing metadata | Continue without temporal features |
| Embedding failure | Use zero vector with warning |
| Model error | Return error response, log details |

## Data Privacy

- Student IDs: Anonymous hashing
- Complaint text: Stored securely
- Metadata: Minimal PII
- Logs: No raw complaint text

## Monitoring Points

Track these metrics at each stage:

1. **Ingestion**: Complaint rate, queue depth
2. **Preprocessing**: Error rate, processing time
3. **Embedding**: Cache hit rate
4. **Prediction**: Latency, confidence distribution
5. **Output**: Urgency distribution, accuracy (if labeled)
