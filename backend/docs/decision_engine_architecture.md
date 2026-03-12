# Decision Engine Architecture

## Overview

The Decision Engine is a modular, production-ready system for computing cluster-level urgency from complaint data. It is designed to be **clustering-source agnostic**, working seamlessly with any clustering mechanism (FAISS, VectorDB, or custom solutions).

## Core Design Principles

### 1. Separation of Concerns

Each component has a single, well-defined responsibility:

- **Aggregation**: Converts complaint-level urgency → cluster-level structural urgency
- **Temporal**: Analyzes arrival patterns → temporal urgency
- **Calibration**: Applies bias correction and smoothing → final urgency
- **Summarization**: Generates human-readable summaries
- **Orchestration**: Coordinates the complete pipeline

### 2. Clustering-Source Independence

**Critical Architectural Decision**: The engine never directly depends on FAISS or VectorDB.

Instead, it operates on a standardized `Complaint` data structure:

```python
@dataclass
class Complaint:
    complaint_id: str
    cluster_id: str      # Provided by external clustering
    urgency_score: float
    timeline_score: float
    timestamp: datetime
    text: str
```

Any clustering source must provide data in this format via the `ComplaintDataSource` protocol.

### 3. Configurability

All weights, thresholds, and hyperparameters are externalized to `config.py`:

- Aggregation weights (α, β, δ)
- Temporal parameters (θ₁, θ₂)
- Fusion weights (λ₁, λ₂)
- Calibration smoothing factor (γ)

This enables tuning without code changes.

## System Architecture

### High-Level Flow

```
Input: List[Complaint] with cluster_id assignments
   ↓
Group by cluster_id
   ↓
For each cluster:
   ├─→ Structural Aggregation  →  U_struct
   ├─→ Temporal Analysis       →  T_k
   └─→ Raw Fusion             →  U_raw = λ₁·U_struct + λ₂·T_k
        ↓
      Calibration:
        ├─→ Size Normalization  →  U_size = U_raw / (1 + log(1 + n))
        └─→ Temporal Smoothing  →  U_final = γ·U_prev + (1-γ)·U_size
        ↓
      Summarization             →  ClusterSummary
   ↓
Output: Ranked List[ClusterSummary]
```

### Layer Architecture

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│         (main.py)                       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         Orchestrator                    │
│  Coordinates complete pipeline          │
└───┬─────────┬─────────┬─────────┬───────┘
    │         │         │         │
┌───▼───┐ ┌──▼───┐ ┌───▼───┐ ┌───▼────┐
│Aggre- │ │Temp- │ │Cali-  │ │Summa-  │
│gation │ │oral  │ │bration│ │rizer   │
└───┬───┘ └──┬───┘ └───┬───┘ └───┬────┘
    └────────┴─────────┴─────────┘
                 │
         ┌───────▼───────┐
         │ Data Interface│
         │   (Protocol)  │
         └───────┬───────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌────▼────┐  ┌────▼─────┐
│FAISS  │  │VectorDB │  │In-Memory │
│Adapter│  │Adapter  │  │Store     │
└───────┘  └─────────┘  └──────────┘
```

## Component Details

### 1. Data Interface Layer (`services/data_interface.py`)

**Purpose**: Abstract clustering source

**Key Classes**:
- `Complaint`: Universal data structure
- `ComplaintDataSource`: Protocol defining required interface
- `InMemoryComplaintDataSource`: Reference implementation

**Benefits**:
- Decouples engine from clustering mechanism
- Enables easy testing (in-memory mock)
- Supports future clustering sources without engine changes

### 2. Aggregation Engine (`engine/aggregation.py`)

**Purpose**: Compute structural urgency from complaint urgencies

**Formula**:
```
U_struct = α·mean + β·max + δ·percentile_90
```

**Responsibility**:
- Extract urgency scores
- Compute statistical measures
- Combine with configured weights

**Statefulness**: Stateless (pure function)

### 3. Temporal Engine (`engine/temporal.py`)

**Purpose**: Analyze complaint arrival patterns

**Metrics**:
- Volume ratio: R = V_recent / (V_historical + ε)
- Arrival rate: λ = 1 / mean_inter_arrival_time
- Temporal intensity: T = σ(θ₁·R + θ₂·λ)

**Responsibility**:
- Detect bursts
- Measure frequency
- Apply sigmoid transformation

**Statefulness**: Stateless (operates on timestamps)

### 4. Calibration Engine (`engine/calibration.py`)

**Purpose**: Correct bias and reduce volatility

**Transformations**:
1. Size normalization: U_size = U_raw / (1 + log(1 + n))
2. Temporal smoothing: U_final = γ·U_prev + (1-γ)·U_size

**Responsibility**:
- Prevent large clusters from dominating
- Smooth urgency changes over time
- Maintain memory of previous urgencies

**Statefulness**: Stateful (maintains memory cache)

### 5. Summarizer (`engine/summarizer.py`)

**Purpose**: Generate human-readable cluster summaries

**Approach**: Extraction-based summarization

**Responsibility**:
- Select top-k urgent complaints
- Concatenate text
- Provide structured output

**Extensibility**: Hooks for abstractive summarization models

### 6. Orchestrator (`engine/orchestrator.py`)

**Purpose**: Coordinate the complete pipeline

**Responsibility**:
- Group complaints by cluster
- Execute all components in sequence
- Build comprehensive response objects
- Handle errors gracefully

**Statefulness**: Delegates to components

## Data Flow

### Complaint-Level to Cluster-Level Mapping

```
Individual Complaints (complaint_id, urgency_score, timestamp)
   ↓
Group by cluster_id
   ↓
Cluster = {C₁, C₂, ..., Cₙ} all sharing same cluster_id
   ↓
Aggregation: [U₁, U₂, ..., Uₙ] → U_struct
Temporal: [t₁, t₂, ..., tₙ] → T_k
   ↓
Cluster Urgency: U_final
```

### State Management

- **Aggregation**: No state (pure function)
- **Temporal**: No persistent state (uses timestamps)
- **Calibration**: Memory of previous urgencies (in-memory cache)
- **Orchestrator**: Stateless (fresh computation each call)

For production deployment with multiple instances, calibration memory should be externalized to Redis or similar.

## Extensibility Points

### Adding New Clustering Sources

1. Implement `ComplaintDataSource` protocol
2. Transform clustering output → `List[Complaint]`
3. No engine code changes required

Example:
```python
class MyClusteringAdapter:
    def fetch_all_complaints(self) -> List[Complaint]:
        # Query your clustering system
        # Transform to Complaint objects
        # Return
        pass
```

### Adding New Urgency Components

To add a new urgency dimension (e.g., sentiment):

1. Create new engine module (e.g., `engine/sentiment.py`)
2. Compute sentiment score per cluster
3. Update fusion in orchestrator to include new component
4. Update config with new fusion weights

### Replacing Summarization

The summarizer is intentionally model-agnostic:

```python
# Current: Extraction-based
summary = "| ".join(top_k_complaints)

# Future: Abstractive with T5
summary = t5_model.generate(top_k_complaints)
```

## Deployment Considerations

### Horizontal Scaling

- Orchestrator is stateless
- Calibration memory needs shared storage (Redis)
- API can be load-balanced

### Performance

- Batch processing: Use `orchestrator.process_complaints()` with all clusters
- Caching: API caches last processed results
- Database: For production, replace in-memory store with persistent DB

### Monitoring

Key metrics to track:
- Request latency per cluster count
- Urgency distribution (prevent saturation)
- Memory growth (calibration cache)
- Error rates

## Testing Strategy

Three test levels:

1. **Unit Tests**: Each component in isolation
   - `test_aggregation.py`
   - `test_temporal.py`
   - `test_calibration.py`

2. **Integration Tests**: Complete pipeline
   - `test_integration.py`
   - Validate end-to-end behavior

3. **API Tests**: HTTP endpoints (future)

## Security Considerations

- Input validation: All scores validated in [0, 1]
- SQL injection: N/A (no direct DB queries)
- Rate limiting: Should be added at API layer
- Authentication: Should be added for production

## Future Enhancements

1. **Explainability**: Add feature importance scores
2. **Feedback Loop**: Incorporate user corrections
3. **Multi-objective**: Optimize for multiple criteria
4. **Real-time**: Convert to streaming pipeline
5. **ML Integration**: Learn weights from data

## Conclusion

The Decision Engine provides a robust, modular foundation for complaint intelligence. Its clustering-source agnostic design ensures flexibility, while comprehensive testing ensures reliability. The architecture supports future enhancements without requiring fundamental changes.
