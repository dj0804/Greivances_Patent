# Decision Logic Documentation

## Overview

The Decision Engine is the **critical fusion layer** that differentiates this system from simple text classifiers. It implements a **multi-signal aggregation framework** that combines machine learning predictions with contextual intelligence—specifically cluster density, temporal frequency patterns, historical resolution feedback, and semantic similarity—to produce robust, adaptive priority scores.

**Core Principle**: Urgency is not an intrinsic property of complaint text alone; it emerges from the **interaction** between content, context, timing, and historical patterns.

---

## Why Urgency Cannot Be Purely Model-Based

### Limitations of Pure ML Classification

**Problem 1: Context Blindness**
```
Complaint A: "AC not working" (filed at 2 AM, 95°F outside, 20 similar complaints today)
Complaint B: "AC not working" (filed at 10 AM, 68°F outside, first complaint this month)

A pure ML model (CNN-BiLSTM) sees identical text → produces ~same urgency score
But actual urgency differs significantly due to:
  - Time (2 AM vs 10 AM)
  - Environmental context (temperature)
  - Systemic issue indicator (20 vs 1 complaints)
```

**Problem 2: Burst Insensitivity**
```
Scenario: Food poisoning outbreak in hostel dining hall
  - 30 complaints in 2 hours about "stomach pain", "vomiting"
  - Individual complaints might score Medium urgency
  - But cluster density reveals systemic health emergency → Critical priority
```

**Problem 3: Historical Drift**
```
Month 1: "Water leak" complaints resolve in 4 hours → Medium urgency
Month 3: Same complaints now take 12 hours → Underestimated urgency

Pure ML model doesn't adapt unless retrained.
Decision engine auto-calibrates based on resolution feedback.
```

**Solution**: **Hybrid Decision Architecture** that fuses ML predictions with rule-based contextual adjustments.

---

## Mathematical Intuition Behind Urgency Aggregation

### Base Formula

```
Priority(complaint) = α·U_model + β·D_cluster + γ·T_temporal + δ·H_history + ε·S_semantic

where:
  U_model    ∈ [0, 1]  : CNN-BiLSTM urgency prediction
  D_cluster  ∈ [0, 1]  : Normalized cluster density score
  T_temporal ∈ [0, 1]  : Temporal urgency boost
  H_history  ∈ [0, 1]  : Historical urgency calibration
  S_semantic ∈ [0, 1]  : Similarity to past high-priority cases
  
  α + β + γ + δ + ε = 1.0  (normalized weights)
```

### Default Weight Configuration

| Weight | Value | Justification |
|--------|-------|---------------|
| α (model) | 0.40 | Primary signal from trained classifier |
| β (cluster) | 0.25 | High weight for systemic issue detection |
| γ (temporal) | 0.15 | Context-aware boost for time-sensitive cases |
| δ (history) | 0.10 | Gradual adaptation to resolution patterns |
| ε (semantic) | 0.10 | Learn from similar past critical cases |

**Rationale**: Model prediction is most reliable (40%), but cluster density (25%) carries significant weight to detect outbreaks/systemic failures that individual complaints might understate.

---

## Signal Components: Detailed Calculation

### 1. Model Urgency Score (U_model)

**Source**: CNN-BiLSTM softmax output

**Calculation**:
```
Model Output: P(Low), P(Medium), P(High)
Example: [0.10, 0.25, 0.65]

U_model = 0.0 × P(Low) + 0.5 × P(Medium) + 1.0 × P(High)
        = 0.0 × 0.10 + 0.5 × 0.25 + 1.0 × 0.65
        = 0.775
```

**Interpretation**: 
- U_model < 0.3 → Low urgency
- 0.3 ≤ U_model < 0.7 → Medium urgency
- U_model ≥ 0.7 → High urgency

---

### 2. Cluster Density Score (D_cluster)

**Purpose**: Detect systemic issues by identifying complaint clusters with high spatial and temporal density.

#### Clustering Approach: HDBSCAN

**Why HDBSCAN?**
| Feature | K-Means | DBSCAN | HDBSCAN |
|---------|---------|--------|---------|
| No. of clusters | Manual | Manual (ε param) | **Auto-detected** |
| Noise handling | Poor | Good | **Excellent** |
| Variable density | No | No | **Yes** |
| Hierarchical | No | No | **Yes** |

HDBSCAN automatically identifies:
- **Core points**: Complaints in dense regions (systemic issues)
- **Outliers**: Isolated complaints (unique issues)
- **Cluster hierarchy**: Emergent patterns at multiple scales

#### Density Calculation

**Step 1: Spatial Density**
```python
def spatial_density(cluster_id, embeddings):
    cluster_vectors = embeddings[labels == cluster_id]
    
    # Pairwise cosine distances
    distances = pairwise_distances(cluster_vectors, metric='cosine')
    
    # Average inter-complaint distance (lower = denser)
    avg_distance = np.mean(distances)
    
    # Invert to get density (higher = denser)
    spatial_dens = 1 / (avg_distance + 1e-6)
    
    return spatial_dens
```

**Intuition**: If all complaints in a cluster have very similar semantic embeddings, they describe the same underlying issue → high density.

**Step 2: Temporal Density**
```python
def temporal_density(cluster_id, timestamps):
    cluster_times = timestamps[labels == cluster_id]
    
    # Time span of cluster (hours)
    time_span = (max(cluster_times) - min(cluster_times)).total_seconds() / 3600
    
    # Complaints per hour
    temporal_dens = len(cluster_times) / max(time_span, 1.0)
    
    return temporal_dens
```

**Intuition**: 20 complaints in 2 hours >> 20 complaints in 2 weeks → burst pattern indicates urgency.

**Step 3: Combined Density Score**
```python
D_spatial = spatial_density(cluster_id, embeddings)
D_temporal = temporal_density(cluster_id, timestamps)

# Weighted combination
D_combined = 0.6 × normalize(D_spatial) + 0.4 × normalize(D_temporal)

# Normalize to [0, 1]
D_cluster = min(D_combined / D_max, 1.0)
```

#### Edge Case: Outliers

```
If complaint is classified as outlier (cluster_id = -1):
    D_cluster = 0.0  (no systemic issue)
    
This is correct behavior:
    - Unique complaint → rely more on model prediction
    - No cluster density boost
```

**Example**:
```
Cluster #7: Water heater failures
  - 15 complaints in last 24 hours
  - Avg cosine distance: 0.15 (very similar)
  
  D_spatial = 1 / 0.15 = 6.67 → normalized to 0.85
  D_temporal = 15 / 24 = 0.625 → normalized to 0.62
  
  D_cluster = 0.6 × 0.85 + 0.4 × 0.62 = 0.51 + 0.25 = 0.76
```

---

### 3. Temporal Urgency Boost (T_temporal)

**Purpose**: Adjust urgency based on **when** the complaint is filed.

#### Time-Based Rules

**Rule 1: Night Hours (12 AM - 6 AM)**
```
is_night = (0 ≤ hour < 6)
Boost = 0.15

Rationale:
  - Limited staff availability
  - Safety concerns heightened
  - Student well-being critical during sleep hours
```

**Example**:
```
2 AM complaint: "Door lock broken"
Base urgency might be Medium (0.5)
Night boost: +0.15 → 0.65 (High)
Reason: Security vulnerability during night is more severe
```

**Rule 2: Peak Hours (6 PM - 12 AM or 7 AM - 10 AM)**
```
is_peak = (19 ≤ hour ≤ 23) or (7 ≤ hour < 10)
Boost = 0.10

Rationale:
  - Students returning from classes (evening)
  - Morning preparation for classes
  - High activity, high impact of failures
```

**Rule 3: Weekend**
```
is_weekend = (day_of_week in [5, 6])  # Saturday, Sunday
Boost = 0.05

Rationale:
  - Reduced maintenance staff
  - Students present in hostel all day (higher exposure to issues)
```

#### Temporal Frequency Pattern

**Burst Detection**:
```python
def calculate_temporal_boost(complaint, temporal_features):
    base_boost = 0.0
    
    # Time-of-day boost
    if temporal_features['is_night']:
        base_boost += 0.15
    elif temporal_features['is_peak_hour']:
        base_boost += 0.10
    
    # Weekend boost
    if temporal_features['is_weekend']:
        base_boost += 0.05
    
    # Frequency spike boost
    complaints_24h = temporal_features['complaints_last_24h']
    complaints_avg = temporal_features['historical_avg_24h']
    
    if complaints_24h > 2 * complaints_avg:  # 2x spike
        base_boost += 0.20  # Burst detected
    elif complaints_24h > 1.5 * complaints_avg:  # 1.5x spike
        base_boost += 0.10  # Elevated frequency
    
    # Cap at 1.0
    T_temporal = min(base_boost, 1.0)
    
    return T_temporal
```

**Example**:
```
Normal week: 5 complaints/day from Hostel H-101
Today: 15 complaints/day (3x spike)

T_temporal = 0.10 (peak hour) + 0.20 (burst) = 0.30

This boosts all complaints from H-101 today, indicating systemic/facility issue.
```

---

### 4. Historical Urgency Calibration (H_history)

**Purpose**: Adapt urgency scoring based on **actual resolution outcomes**.

#### Feedback Loop Mechanism

**Data Tracked**:
```python
resolution_database = {
    "complaint_id": "C_001",
    "predicted_urgency": "High",
    "actual_resolution_time": 8.5,  # hours
    "target_resolution_time": 2.0,  # for High urgency
    "satisfaction_score": 3.2,  # out of 5
    "was_escalated": True
}
```

**Calibration Logic**:
```python
def historical_calibration(complaint_text, predicted_urgency):
    # Find similar past complaints (semantic search)
    similar_cases = vector_store.search(complaint_embedding, top_k=10)
    
    # Extract resolution times
    resolution_times = [case['resolution_time'] for case in similar_cases]
    avg_resolution_time = np.mean(resolution_times)
    
    # Expected SLA for predicted urgency
    sla_targets = {"Low": 48, "Medium": 8, "High": 2}  # hours
    expected_sla = sla_targets[predicted_urgency]
    
    # If similar complaints historically took longer → increase urgency
    if avg_resolution_time > 1.5 * expected_sla:
        H_history = 0.80  # Upward calibration
    elif avg_resolution_time > expected_sla:
        H_history = 0.60  # Moderate calibration
    elif avg_resolution_time < 0.8 * expected_sla:
        H_history = 0.30  # Downward calibration (over-predicted)
    else:
        H_history = 0.50  # No adjustment
    
    return H_history
```

**Example**:
```
Complaint: "WiFi not working"
Predicted: Medium (SLA = 8 hours)

Historical data: Past WiFi complaints took avg 14 hours to resolve
  → avg_resolution_time (14) > 1.5 × SLA (12)
  → H_history = 0.80
  → System learns to prioritize WiFi complaints higher

After 3 months: WiFi infrastructure improved, now resolves in 4 hours
  → avg_resolution_time (4) < 0.8 × SLA (6.4)
  → H_history = 0.30
  → System auto-adjusts downward
```

**This creates a self-correcting feedback loop** without manual retraining.

---

### 5. Semantic Similarity to Critical Cases (S_semantic)

**Purpose**: Learn from past critical/emergency complaints.

#### Process

**Step 1: Maintain Critical Case Database**
```python
critical_cases = [
    {"id": "C_457", "text": "Fire alarm not working", "urgency": "Critical"},
    {"id": "C_892", "text": "Gas leak smell in kitchen", "urgency": "Critical"},
    {"id": "C_1203", "text": "Student injured, elevator stuck", "urgency": "Critical"},
    ...
]
```

**Step 2: Compute Similarity**
```python
def semantic_similarity_boost(complaint_embedding):
    # Get embeddings of all critical cases
    critical_embeddings = [case['embedding'] for case in critical_cases]
    
    # Cosine similarity to each critical case
    similarities = cosine_similarity(complaint_embedding, critical_embeddings)
    
    # Max similarity (most similar critical case)
    max_similarity = np.max(similarities)
    
    S_semantic = max_similarity  # Already in [0, 1]
    
    return S_semantic
```

**Example**:
```
New complaint: "Smell of burning wires in room 305"

Similarity to critical cases:
  - "Fire alarm not working": 0.45
  - "Gas leak smell": 0.72  ← High similarity
  - "Elevator stuck": 0.23
  
S_semantic = 0.72

Even if model predicts Medium (U_model = 0.50), the high semantic similarity to past gas leak (critical) boosts final priority.
```

---

## Integrated Decision Formula: Complete Example

**Scenario**: Real complaint processing

```
COMPLAINT:
  Text: "The water heater in our block has been broken since this morning. No hot water in any room. This is the third complaint this week!!!"
  Timestamp: 2026-02-17 08:30:00 (Monday, 8:30 AM)
  Hostel: H-101

STEP 1: Preprocessing
  Cleaned: "[Multiple Exclamations] the water heater in our block has been broken since this morning no hot water in any room this is the third complaint this week"
  Tokens: ["water", "heater", "block", "broken", "morning", "hot", "water", "room", "third", "complaint", "week"]

STEP 2: Model Prediction (CNN-BiLSTM)
  P(Low) = 0.12, P(Medium) = 0.28, P(High) = 0.60
  U_model = 0.0 × 0.12 + 0.5 × 0.28 + 1.0 × 0.60 = 0.74

STEP 3: Cluster Density
  Semantic embedding → Search vector DB
  Found cluster #7: "Water heater failures" with 18 complaints in last 3 days
  
  Spatial density: avg_distance = 0.18 → spatial_dens = 5.56 → normalized = 0.82
  Temporal density: 18 complaints / 72 hours = 0.25 → normalized = 0.58
  
  D_cluster = 0.6 × 0.82 + 0.4 × 0.58 = 0.492 + 0.232 = 0.72

STEP 4: Temporal Analysis
  Hour: 8 (morning)
  is_peak_hour: True (7-10 AM)
  is_weekend: False
  
  Complaints from H-101 in last 24h: 8
  Historical avg: 3
  Spike ratio: 8 / 3 = 2.67 → Burst detected
  
  T_temporal = 0.10 (peak) + 0.20 (burst) = 0.30

STEP 5: Historical Calibration
  Similar past water heater complaints (top 5):
    Resolution times: [3.2, 4.5, 2.8, 5.1, 3.9] hours
    avg_resolution_time = 3.9 hours
  
  Predicted urgency: High (SLA = 2 hours)
  3.9 > 1.5 × 2.0 = 3.0 → Upward calibration
  
  H_history = 0.75

STEP 6: Semantic Similarity
  Max similarity to critical cases: 0.35 (moderate)
  S_semantic = 0.35

STEP 7: Fusion
  Priority = α·U_model + β·D_cluster + γ·T_temporal + δ·H_history + ε·S_semantic
          = 0.40 × 0.74 + 0.25 × 0.72 + 0.15 × 0.30 + 0.10 × 0.75 + 0.10 × 0.35
          = 0.296 + 0.180 + 0.045 + 0.075 + 0.035
          = 0.631

STEP 8: Routing Decision
  Priority = 0.631 < 0.65
  BUT: Cluster density (0.72) and burst pattern detected
  
  Override rule: If D_cluster > 0.70 AND T_temporal > 0.25 → Escalate
  
  FINAL DECISION:
    Priority Score: 0.631 (with escalation flag)
    Urgency Level: High
    Routing: Priority Maintenance Queue + Notify Facilities Manager
    SLA: 2 hours
    Cluster: #7 (Water Heater Failures - systemic issue)
    Explanation: "Systemic issue detected: 18 similar complaints in 3 days from cluster #7. Burst pattern identified (8 complaints in last 24h vs avg 3). Historical data shows water heater issues take 3.9h vs 2h SLA. Escalated to facilities management for infrastructure review."
```

---

## Edge Cases and Special Handling

### Case 1: Single Complaint vs. Burst Complaints

**Scenario A: Isolated Complaint**
```
Complaint: "Light bulb in corridor broken"
Cluster: Outlier (cluster_id = -1)

D_cluster = 0.0 (no cluster)
T_temporal = 0.05 (weekend)
U_model = 0.30 (Low)
H_history = 0.40
S_semantic = 0.15

Priority = 0.40×0.30 + 0.25×0.0 + 0.15×0.05 + 0.10×0.40 + 0.10×0.15
         = 0.120 + 0.0 + 0.0075 + 0.040 + 0.015
         = 0.1825 → Low urgency

Routing: Standard Queue, SLA 48 hours
```

**Scenario B: Burst Pattern**
```
Complaint: "Light bulbs in corridor broken" (20th complaint today)
Cluster: #12 (Electrical issues, 20 complaints in 6 hours)

D_cluster = 0.85 (dense cluster, rapid growth)
T_temporal = 0.25 (burst + peak hour)
U_model = 0.35 (individually Low)
H_history = 0.50
S_semantic = 0.20

Priority = 0.40×0.35 + 0.25×0.85 + 0.15×0.25 + 0.10×0.50 + 0.10×0.20
         = 0.140 + 0.2125 + 0.0375 + 0.050 + 0.020
         = 0.460 → Medium urgency

Routing: Priority Queue + Flag for electrician (batch resolution)
Explanation: "20 similar complaints suggest electrical circuit failure, not individual bulbs"
```

**Key Insight**: Cluster density transforms low-urgency individual complaints into medium/high priority systemic issues.

---

### Case 2: Night Emergency Detection

**Complaint**: "Heard loud noise from roof, pieces falling"
**Time**: 2:30 AM

```
U_model = 0.68 (High - model detects urgency from "falling pieces")
D_cluster = 0.0 (outlier - unique complaint)
T_temporal = 0.15 (night boost)
H_history = 0.65 (structural issues historically critical)
S_semantic = 0.55 (similar to past emergency cases)

Priority = 0.40×0.68 + 0.25×0.0 + 0.15×0.15 + 0.10×0.65 + 0.10×0.55
         = 0.272 + 0.0 + 0.0225 + 0.065 + 0.055
         = 0.4145

BUT: Night emergency override rule
IF (is_night AND (U_model > 0.6 OR S_semantic > 0.5)):
    Priority = min(Priority × 1.5, 1.0)
    
Priority = 0.4145 × 1.5 = 0.622 → High urgency

Routing: Emergency Response Team
SLA: 30 minutes (safety priority)
```

---

### Case 3: False Positive Mitigation

**Problem**: Student submits "URGENT HELP NEEDED" for minor issue (e.g., "WiFi slow in my room")

```
U_model = 0.72 (High - model detects "URGENT" keyword and caps)
D_cluster = 0.15 (few similar complaints)
T_temporal = 0.10 (peak hour)
H_history = 0.25 (WiFi speed issues historically low priority)
S_semantic = 0.20 (low similarity to critical cases)

Priority = 0.40×0.72 + 0.25×0.15 + 0.15×0.10 + 0.10×0.25 + 0.10×0.20
         = 0.288 + 0.0375 + 0.015 + 0.025 + 0.020
         = 0.3855 → Medium urgency (downgraded from High)

Explanation: "Despite urgent language, low cluster density (0.15), low historical urgency (0.25), and low similarity to critical cases (0.20) suggest individual preference rather than systemic issue."
```

**This demonstrates robustness**: The multi-signal approach prevents keyword gaming.

---

## Adaptive Weight Tuning (Advanced)

**Dynamic Weight Adjustment** based on system state:

### Normal Operation
```python
weights = {
    'α': 0.40,  # Model
    'β': 0.25,  # Cluster
    'γ': 0.15,  # Temporal
    'δ': 0.10,  # History
    'ε': 0.10   # Semantic
}
```

### Outbreak/Burst Mode
```
IF global_complaint_rate > 2 × historical_avg:
    # Increase cluster weight during outbreak
    weights = {
        'α': 0.30,  # Reduce model weight
        'β': 0.40,  # Increase cluster weight
        'γ': 0.15,
        'δ': 0.08,
        'ε': 0.07
    }
```

### Model Uncertainty Mode
```
IF U_model_confidence < 0.6:  # Low confidence prediction
    # Rely more on semantic similarity to known cases
    weights = {
        'α': 0.30,
        'β': 0.25,
        'γ': 0.15,
        'δ': 0.10,
        'ε': 0.20   # Increase semantic weight
    }
```

---

## Patent-Worthy Novel Aspects

### 1. Multi-Signal Fusion Framework
**Innovation**: First system to jointly optimize urgency scoring using 5 heterogeneous signals (ML, density, temporal, historical, semantic) rather than single-model classification.

### 2. Density-Aware Prioritization
**Innovation**: HDBSCAN cluster density as a systemic issue detector. Transforms low-urgency individual complaints into high-priority alerts when spatial+temporal density exceeds thresholds.

### 3. Temporal Frequency Burst Detection
**Innovation**: Real-time complaint frequency analysis to detect outbreak patterns, automatically boosting urgency during rapid complaint growth.

### 4. Self-Calibrating Historical Feedback
**Innovation**: Continuous online learning from resolution outcomes without model retraining. Automatically adjusts urgency thresholds based on actual SLA performance.

### 5. Semantic Precedent Matching
**Innovation**: Case-based reasoning using vector similarity to past critical complaints. Enables zero-shot detection of novel emergencies similar to known patterns.

### 6. Adaptive Context-Aware Routing
**Innovation**: Routing decisions consider not just urgency level but also cluster membership (systemic → facilities management vs individual → maintenance team).

---

## Routing Logic Details

**Post-Priority Decision Tree**:

```
Priority Score Calculated
        │
        ▼
┌───────────────────────────────────┐
│ Is Priority > 0.85?               │
│ OR (is_night AND Priority > 0.60) │
└───────┬───────────────────────────┘
        │
    ┌───┴───┐
   YES      NO
    │       │
    ▼       ▼
┌────────┐  ┌─────────────────────────┐
│CRITICAL│  │ Is Priority > 0.65?     │
│Route to│  │ OR Cluster Density>0.70?│
│Emergency│  └───────┬─────────────────┘
│Response│          │
│SLA:30min│     ┌───┴───┐
└────────┘     YES      NO
               │        │
               ▼        ▼
            ┌────┐   ┌────────────────────┐
            │HIGH│   │ Is Cluster ID valid?│
            │Priority│└───────┬────────────┘
            │Queue  │        │
            │SLA:2h │    ┌───┴───┐
            └────┘  YES      NO
                    │        │
                    ▼        ▼
                ┌────────┐ ┌─────┐
                │SYSTEMIC│ │STANDARD│
                │Batch   │ │Queue  │
                │Resolve │ │SLA:48h│
                │SLA:24h │ └─────┘
                └────────┘
```

**Routing Metadata** (attached to routed complaint):
```json
{
  "route": {
    "queue": "Priority Maintenance",
    "team": "Electrical Team",
    "sla_hours": 2,
    "reason": "Systemic issue - cluster #12",
    "related_complaints": ["C_001", "C_034", "C_089"],
    "batch_resolution_eligible": true
  }
}
```

---

## Summary: Decision Logic Flow

```
1. Receive preprocessed complaint + metadata
       ↓
2. Extract 5 signals:
   - U_model (CNN-BiLSTM)
   - D_cluster (HDBSCAN density)
   - T_temporal (time-based boost + burst detection)
   - H_history (resolution feedback calibration)
   - S_semantic (similarity to critical cases)
       ↓
3. Apply weighted fusion formula
       ↓
4. Check override rules:
   - Night emergency escalation
   - Burst pattern escalation
   - Cluster density escalation
       ↓
5. Generate final priority score [0, 1]
       ↓
6. Route to appropriate queue/team
       ↓
7. Log decision + explanations for transparency
       ↓
8. Track resolution outcome for feedback loop
```

**This decision architecture ensures**:
- **Robustness**: Not dependent on single signal
- **Adaptability**: Self-correcting via historical feedback
- **Transparency**: Explainable decisions with contributory factors
- **Context-awareness**: Considers time, patterns, history
- **Systemic detection**: Identifies outbreaks and infrastructure failures
| flooded | Water damage | +0.20 |
| injured | Medical concern | +0.30 |
| fire | Safety hazard | +0.30 |
| smoke | Safety hazard | +0.25 |
| gas leak | Safety hazard | +0.30 |

**Multiple Keywords**: Stacking boost (capped at +0.30)

**Example**:
```
"Emergency! Water pipe broken and flooding room"
Keywords: emergency (+0.20), broken (+0.10), flooding (+0.20)
Base: 0.50 → Adjusted: 0.90 (capped at 1.0)
```

### 4. Historical Context

Repeated complaints about the same issue indicate systemic problems.

#### Boost Calculation

```python
if complaint_history_count > 0:
    boost = 0.20
    reason = f"Repeated complaint (count: {complaint_history_count})"
```

**Matching Criteria**:
- Same hostel
- Same complaint type (e.g., "plumbing")
- Within last 30 days
- Not yet resolved

**Example**:
```
3rd complaint about "AC not cooling" in 2 weeks
Base: 0.45 → Adjusted: 0.65
Reason: "Repeated complaint (count: 2)"
```

### 5. Emotional Indicators

Preserved from preprocessing stage, these influence base model prediction.

| Indicator | Signal | Impact |
|-----------|--------|--------|
| [CAPS_ON] | Shouting/stress | Model learns urgency |
| [Multiple Exclamations] | Frustration | Model learns urgency |
| [Impatient Questioning] | Impatience | Model learns urgency |
| [Emphasis] | Strong emotion | Model learns urgency |

**Note**: These are learned features, not explicit boosts.

## Decision Algorithm

### Pseudocode

```python
def make_decision(model_prediction, temporal_features, complaint_history):
    # Start with base prediction
    base_urgency = model_prediction.urgency_level
    base_confidence = model_prediction.confidence
    
    # Initialize boost
    total_boost = 0.0
    adjustments = []
    
    # 1. Temporal adjustments
    if temporal_features.is_night:
        total_boost += 0.15
        adjustments.append({
            "type": "temporal",
            "boost": 0.15,
            "reason": "Night-time complaint"
        })
    
    if temporal_features.is_peak_hour and not temporal_features.is_night:
        total_boost += 0.10
        adjustments.append({
            "type": "temporal", 
            "boost": 0.10,
            "reason": "Peak hour complaint"
        })
    
    if temporal_features.is_weekend:
        total_boost += 0.05
        adjustments.append({
            "type": "temporal",
            "boost": 0.05,
            "reason": "Weekend complaint"
        })
    
    # 2. Keyword analysis
    text = model_prediction.cleaned_text.lower()
    keyword_boost = 0.0
    
    critical_keywords = ["emergency", "urgent", "immediately", ...]
    found_keywords = [kw for kw in critical_keywords if kw in text]
    
    if found_keywords:
        keyword_boost = min(0.30, len(found_keywords) * 0.10)
        total_boost += keyword_boost
        adjustments.append({
            "type": "keyword",
            "boost": keyword_boost,
            "reason": f"Keywords: {', '.join(found_keywords)}"
        })
    
    # 3. History analysis
    if len(complaint_history) > 0:
        history_boost = 0.20
        total_boost += history_boost
        adjustments.append({
            "type": "history",
            "boost": history_boost,
            "reason": f"Repeated complaint (count: {len(complaint_history)})"
        })
    
    # Calculate final confidence
    adjusted_confidence = min(1.0, base_confidence + total_boost)
    
    # Determine final urgency level
    if adjusted_confidence >= 0.80:
        final_urgency = "High"
    elif adjusted_confidence >= 0.50:
        final_urgency = "Medium"
    else:
        # Don't downgrade if base was already high
        if base_urgency == "High":
            final_urgency = "Medium"
        else:
            final_urgency = "Low"
    
    return {
        "urgency_level": final_urgency,
        "confidence": adjusted_confidence,
        "original_urgency": base_urgency,
        "original_confidence": base_confidence,
        "adjustments": adjustments,
        "total_boost": total_boost,
        "requires_immediate_action": adjusted_confidence >= 0.70
    }
```

## Escalation Thresholds

### Confidence Thresholds

| Threshold | Action | Response Time |
|-----------|--------|---------------|
| ≥ 0.80 | Immediate escalation | < 1 hour |
| 0.50-0.79 | Standard processing | < 24 hours |
| < 0.50 | Normal queue | < 48 hours |

### Override Rules

Certain conditions bypass confidence thresholds:

1. **Safety Keywords** (fire, injured, gas leak)
   - Immediate escalation regardless of confidence
   
2. **Infrastructure Failure** + Night Time
   - elevator, door lock, security → Immediate
   
3. **Multiple Complaints** (≥ 5 students, same issue)
   - Collective urgency → Escalate

## Decision Examples

### Example 1: High Confidence, Night Time

```
Input:
- Text: "AC not working, room very hot"
- Time: 2:30 AM
- History: None

Processing:
- Base: Medium (0.65)
- Night boost: +0.15 → 0.80
- Final: High (0.80)

Reasoning:
- Comfort issue normally medium priority
- Night-time AC failure affects sleep
- Student health/comfort at risk
```

### Example 2: Low Base, Critical Keyword

```
Input:
- Text: "Small emergency with bathroom tap"
- Time: 3:00 PM
- History: None

Processing:
- Base: Low (0.40)
- Keyword "emergency": +0.20 → 0.60
- Final: Medium (0.60)

Reasoning:
- "Small" downweights urgency
- "Emergency" triggers boost
- Balanced to medium priority
```

### Example 3: Repeated Complaint

```
Input:
- Text: "WiFi still not working in room"
- Time: 6:00 PM
- History: 2 previous complaints

Processing:
- Base: Medium (0.55)
- Peak hour: +0.10 → 0.65
- History: +0.20 → 0.85
- Final: High (0.85)

Reasoning:
- Third complaint shows neglect
- Affects student academics
- Peak study time
```

### Example 4: Safety Concern

```
Input:
- Text: "Smoke coming from electrical socket"
- Time: 10:00 AM
- History: None

Processing:
- Base: High (0.75)
- Keyword "smoke": +0.25 → 1.00
- Final: High (1.00)

Reasoning:
- Fire/electrical hazard
- Immediate safety risk
- Maximum urgency assigned
```

## Customization

### Adjusting Boost Values

Edit `src/decision_engine/urgency_logic.py`:

```python
self.escalation_rules = {
    'peak_hour_boost': 0.10,      # Modify this
    'night_boost': 0.15,          # Modify this
    'weekend_boost': 0.05,        # Modify this
    'repeated_complaint_boost': 0.20,  # Modify this
    'critical_keywords': [...],   # Add/remove keywords
}
```

### Adding New Rules

```python
# In UrgencyDecisionEngine class

def _check_custom_rule(self, text, metadata):
    """Add your custom logic here."""
    boost = 0.0
    
    # Example: Boost for specific hostel
    if metadata.get('hostel_id') == 'H-101':
        boost += 0.05
    
    return boost
```

## Monitoring & Tuning

### Metrics to Track

1. **Boost Distribution**:
   - How often are boosts applied?
   - Average boost magnitude?

2. **Override Rate**:
   - How often does decision engine change urgency level?
   - Low → Medium, Medium → High?

3. **False Positives**:
   - High urgency complaints resolved easily
   - Indicates over-escalation

4. **False Negatives**:
   - Low urgency complaints that became critical
   - Indicates under-escalation

### A/B Testing

Compare decision engine ON vs OFF:
- Response times
- User satisfaction
- Staff workload distribution

### Continuous Improvement

1. Review misclassified complaints weekly
2. Adjust boost values based on data
3. Add new keywords as patterns emerge
4. Refine temporal rules based on staff schedules

## Best Practices

1. **Conservative Escalation**: Better to over-prioritize than miss critical issues
2. **Transparent Reasoning**: Always log adjustment reasons
3. **Regular Review**: Update rules based on operational data
4. **Safety First**: Safety-related keywords should have highest boosts
5. **Context Matters**: Same complaint has different urgency at different times
