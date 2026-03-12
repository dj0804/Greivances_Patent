# API Contract Documentation

## Overview

The Decision Engine exposes a RESTful API for processing complaints and retrieving cluster urgency rankings.

**Base URL**: `http://localhost:8000` (development)

**API Version**: 1.0.0

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
3. [Request Models](#request-models)
4. [Response Models](#response-models)
5. [Error Handling](#error-handling)
6. [Examples](#examples)
7. [Rate Limiting](#rate-limiting)

---

## Authentication

**Current**: No authentication (development mode)

**Production**: Should implement:
- API key authentication
- JWT tokens
- OAuth 2.0

Header format (future):
```
Authorization: Bearer <token>
```

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check service health and status.

#### Request

No parameters required.

#### Response

```json
{
  "status": "healthy",
  "service": "decision-engine",
  "version": "1.0.0",
  "timestamp": "2024-02-15T14:30:00Z"
}
```

#### Status Codes

- `200 OK`: Service is healthy

---

### 2. Process Complaints

**POST** `/decision/process`

Process complaints and compute cluster urgencies.

#### Request Body

```json
{
  "complaints": [
    {
      "complaint_id": "CMP-001",
      "cluster_id": "CLUSTER-1",
      "urgency_score": 0.85,
      "timeline_score": 0.75,
      "timestamp": "2024-02-15T14:30:00Z",
      "text": "Critical system outage"
    }
  ],
  "recompute_all": true
}
```

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `complaints` | Array[ComplaintInput] | Yes | List of complaints to process |
| `recompute_all` | Boolean | No | Whether to recompute all clusters (default: true) |

**ComplaintInput** schema:

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `complaint_id` | String | Yes | - | Unique complaint identifier |
| `cluster_id` | String | Yes | - | Cluster assignment |
| `urgency_score` | Float | Yes | [0.0, 1.0] | Complaint urgency score |
| `timeline_score` | Float | Yes | [0.0, 1.0] | Timeline factor |
| `timestamp` | DateTime | Yes | ISO 8601 | Submission timestamp |
| `text` | String | Yes | min_length=1 | Complaint text |

#### Response

```json
{
  "status": "success",
  "clusters_processed": 15,
  "total_complaints": 234,
  "ranked_clusters": [
    {
      "cluster_id": "CLUSTER-42",
      "final_urgency": 0.85,
      "complaint_count": 23,
      "top_complaints": ["CMP-001", "CMP-002", "CMP-003"],
      "summary_text": "Critical outage affecting payment systems...",
      "breakdown": {
        "structural_urgency": 0.82,
        "temporal_urgency": 0.75,
        "raw_urgency": 0.79,
        "size_normalized_urgency": 0.76,
        "final_urgency": 0.85,
        "mean_urgency": 0.78,
        "max_urgency": 0.95,
        "percentile_90_urgency": 0.88,
        "volume_ratio": 2.1,
        "arrival_rate": 3.5,
        "complaint_count": 23,
        "previous_urgency": 0.80
      },
      "earliest_complaint": "2024-02-15T10:00:00Z",
      "latest_complaint": "2024-02-15T16:30:00Z"
    }
  ],
  "processing_timestamp": "2024-02-15T17:00:00Z"
}
```

#### Status Codes

- `200 OK`: Successfully processed
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Processing failed

---

### 3. Get Ranked Clusters

**GET** `/decision/ranked`

Retrieve last computed cluster rankings.

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `top_n` | Integer | No | None | Return only top N clusters |
| `min_urgency_threshold` | Float | No | None | Filter clusters below threshold |

#### Examples

```
GET /decision/ranked
GET /decision/ranked?top_n=10
GET /decision/ranked?min_urgency_threshold=0.5
GET /decision/ranked?top_n=10&min_urgency_threshold=0.6
```

#### Response

```json
{
  "clusters": [
    {
      "cluster_id": "CLUSTER-42",
      "final_urgency": 0.85,
      ...
    }
  ],
  "total_clusters": 15,
  "retrieved_at": "2024-02-15T17:05:00Z"
}
```

#### Status Codes

- `200 OK`: Successfully retrieved
- `404 Not Found`: No processed results available
- `400 Bad Request`: Invalid filter parameters

---

### 4. Clear Cache

**DELETE** `/decision/cache`

Clear cached results and calibration memory.

#### Request

No parameters required.

#### Response

No content (status 204).

#### Status Codes

- `204 No Content`: Successfully cleared

---

### 5. Get Configuration

**GET** `/decision/config`

Retrieve current engine configuration.

#### Response

```json
{
  "aggregation": {
    "alpha": 0.4,
    "beta": 0.3,
    "delta": 0.3,
    "percentile_threshold": 90.0
  },
  "temporal": {
    "theta_1": 2.0,
    "theta_2": 1.5,
    "recent_window_hours": 24.0,
    "historical_window_hours": 168.0
  },
  "calibration": {
    "gamma": 0.7,
    "enable_smoothing": true
  },
  "fusion": {
    "lambda_1": 0.6,
    "lambda_2": 0.4
  }
}
```

#### Status Codes

- `200 OK`: Successfully retrieved

---

## Request Models

### ComplaintInput

```python
{
  "complaint_id": str,        # Required, unique identifier
  "cluster_id": str,          # Required, cluster assignment
  "urgency_score": float,     # Required, range [0.0, 1.0]
  "timeline_score": float,    # Required, range [0.0, 1.0]
  "timestamp": datetime,      # Required, ISO 8601 format
  "text": str                 # Required, min length 1
}
```

### ProcessDecisionRequest

```python
{
  "complaints": List[ComplaintInput],  # Required, min 1 item
  "recompute_all": bool                # Optional, default true
}
```

### GetRankedRequest

```python
{
  "top_n": Optional[int],                    # Optional, >= 1
  "min_urgency_threshold": Optional[float]   # Optional, [0.0, 1.0]
}
```

---

## Response Models

### ClusterUrgencyBreakdown

```python
{
  "structural_urgency": float,       # Aggregated structural component
  "temporal_urgency": float,         # Temporal dynamics component
  "raw_urgency": float,              # Fused before calibration
  "size_normalized_urgency": float,  # After size penalty
  "final_urgency": float,            # Final calibrated score
  "mean_urgency": float,             # Mean of complaints
  "max_urgency": float,              # Maximum urgency
  "percentile_90_urgency": float,    # 90th percentile
  "volume_ratio": float,             # Recent vs historical volume
  "arrival_rate": float,             # Complaints per hour
  "complaint_count": int,            # Number of complaints
  "previous_urgency": Optional[float] # Previous urgency for smoothing
}
```

### ClusterSummary

```python
{
  "cluster_id": str,
  "final_urgency": float,
  "complaint_count": int,
  "top_complaints": List[str],
  "summary_text": str,
  "breakdown": ClusterUrgencyBreakdown,
  "earliest_complaint": datetime,
  "latest_complaint": datetime
}
```

### ProcessDecisionResponse

```python
{
  "status": str,
  "clusters_processed": int,
  "total_complaints": int,
  "ranked_clusters": List[ClusterSummary],  # Sorted by urgency desc
  "processing_timestamp": datetime
}
```

### GetRankedResponse

```python
{
  "clusters": List[ClusterSummary],
  "total_clusters": int,
  "retrieved_at": datetime
}
```

---

## Error Handling

### Error Response Format

All errors return standardized structure:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error description",
  "timestamp": "2024-02-15T17:00:00Z",
  "details": {
    "field": "urgency_score",
    "value": 1.5
  }
}
```

### Common Error Types

| Error Type | Status Code | Cause |
|------------|-------------|-------|
| `ValidationError` | 400 | Invalid input data |
| `NotFoundError` | 404 | Resource not found |
| `ProcessingError` | 500 | Internal processing failure |
| `InternalServerError` | 500 | Unexpected error |

### Validation Errors

**Invalid urgency score**:
```json
{
  "error": "ValidationError",
  "message": "urgency_score must be in [0, 1], got 1.5",
  "details": {
    "field": "urgency_score",
    "value": 1.5
  }
}
```

**Empty complaints list**:
```json
{
  "error": "ValidationError",
  "message": "Cannot process empty complaint list",
  "details": null
}
```

**No processed results**:
```json
{
  "error": "NotFoundError",
  "message": "No processed results available. Call /decision/process first.",
  "details": null
}
```

---

## Examples

### Example 1: Process Small Dataset

**Request**:
```bash
curl -X POST http://localhost:8000/decision/process \
  -H "Content-Type: application/json" \
  -d '{
    "complaints": [
      {
        "complaint_id": "C1",
        "cluster_id": "CLUSTER_A",
        "urgency_score": 0.8,
        "timeline_score": 0.7,
        "timestamp": "2024-02-15T10:00:00Z",
        "text": "Payment system down"
      },
      {
        "complaint_id": "C2",
        "cluster_id": "CLUSTER_A",
        "urgency_score": 0.9,
        "timeline_score": 0.8,
        "timestamp": "2024-02-15T11:00:00Z",
        "text": "Still cannot process payments"
      },
      {
        "complaint_id": "C3",
        "cluster_id": "CLUSTER_B",
        "urgency_score": 0.5,
        "timeline_score": 0.6,
        "timestamp": "2024-02-15T09:00:00Z",
        "text": "Minor UI glitch"
      }
    ],
    "recompute_all": true
  }'
```

**Response**:
```json
{
  "status": "success",
  "clusters_processed": 2,
  "total_complaints": 3,
  "ranked_clusters": [
    {
      "cluster_id": "CLUSTER_A",
      "final_urgency": 0.78,
      "complaint_count": 2,
      "top_complaints": ["C2", "C1"],
      "summary_text": "[C2] Still cannot process payments | [C1] Payment system down",
      "breakdown": {
        "structural_urgency": 0.85,
        "temporal_urgency": 0.68,
        "raw_urgency": 0.78,
        "size_normalized_urgency": 0.56,
        "final_urgency": 0.78,
        "mean_urgency": 0.85,
        "max_urgency": 0.9,
        "percentile_90_urgency": 0.89,
        "volume_ratio": 1.0,
        "arrival_rate": 1.0,
        "complaint_count": 2,
        "previous_urgency": null
      },
      "earliest_complaint": "2024-02-15T10:00:00Z",
      "latest_complaint": "2024-02-15T11:00:00Z"
    },
    {
      "cluster_id": "CLUSTER_B",
      "final_urgency": 0.45,
      "complaint_count": 1,
      ...
    }
  ],
  "processing_timestamp": "2024-02-15T12:00:00Z"
}
```

### Example 2: Retrieve Top 5 Urgent Clusters

**Request**:
```bash
curl -X GET "http://localhost:8000/decision/ranked?top_n=5"
```

**Response**:
```json
{
  "clusters": [
    {
      "cluster_id": "CLUSTER_1",
      "final_urgency": 0.92,
      ...
    },
    {
      "cluster_id": "CLUSTER_5",
      "final_urgency": 0.87,
      ...
    },
    ...
  ],
  "total_clusters": 20,
  "retrieved_at": "2024-02-15T13:00:00Z"
}
```

### Example 3: Filter by Urgency Threshold

**Request**:
```bash
curl -X GET "http://localhost:8000/decision/ranked?min_urgency_threshold=0.7"
```

Returns only clusters with `final_urgency >= 0.7`.

---

## Rate Limiting

**Current**: No rate limiting (development)

**Recommended for Production**:
- 100 requests per minute per API key
- 1000 requests per hour per API key

**Headers** (future):
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1676473200
```

---

## Pagination

**Current**: No pagination (all results returned)

**Recommended for Production**:

For large datasets, implement cursor-based pagination:

```
GET /decision/ranked?limit=20&cursor=eyJjbHVzdGVyX2lkIjoi...
```

Response:
```json
{
  "clusters": [...],
  "pagination": {
    "next_cursor": "eyJjbHVzdGVy...",
    "has_more": true
  }
}
```

---

## Versioning

**Current Version**: 1.0.0

**Future Versions**: Use URL path versioning

```
/v1/decision/process
/v2/decision/process
```

---

## OpenAPI/Swagger Documentation

Interactive API documentation available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Client Libraries

### Python Example

```python
import requests
from datetime import datetime

# Process complaints
response = requests.post(
    "http://localhost:8000/decision/process",
    json={
        "complaints": [
            {
                "complaint_id": "C1",
                "cluster_id": "CLUSTER_A",
                "urgency_score": 0.85,
                "timeline_score": 0.75,
                "timestamp": datetime.now().isoformat(),
                "text": "Critical issue"
            }
        ]
    }
)

results = response.json()
print(f"Top cluster: {results['ranked_clusters'][0]['cluster_id']}")
```

### JavaScript Example

```javascript
const axios = require('axios');

async function processComplaints() {
  const response = await axios.post(
    'http://localhost:8000/decision/process',
    {
      complaints: [
        {
          complaint_id: 'C1',
          cluster_id: 'CLUSTER_A',
          urgency_score: 0.85,
          timeline_score: 0.75,
          timestamp: new Date().toISOString(),
          text: 'Critical issue'
        }
      ]
    }
  );
  
  console.log('Top cluster:', response.data.ranked_clusters[0].cluster_id);
}
```

---

## Best Practices

### 1. Batch Processing

Process multiple complaints in single request for efficiency:

```python
# Good: Single request with 100 complaints
complaints = [...]  # 100 complaints
response = post('/decision/process', json={'complaints': complaints})

# Bad: 100 separate requests
for complaint in complaints:
    post('/decision/process', json={'complaints': [complaint]})
```

### 2. Caching

Use `/decision/ranked` to retrieve cached results:

```python
# Process once
post('/decision/process', json={...})

# Retrieve multiple times without reprocessing
get('/decision/ranked?top_n=10')
get('/decision/ranked?min_urgency_threshold=0.8')
```

### 3. Error Handling

Always handle errors gracefully:

```python
try:
    response = requests.post('/decision/process', json={...})
    response.raise_for_status()
    results = response.json()
except requests.exceptions.HTTPError as e:
    error_data = e.response.json()
    print(f"Error: {error_data['message']}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

---

## Changelog

### Version 1.0.0 (2024-02-15)

- Initial release
- POST `/decision/process`
- GET `/decision/ranked`
- GET `/health`
- DELETE `/decision/cache`
- GET `/decision/config`

---

## Support

For issues or questions:
- GitHub Issues: (link to repo)
- Email: support@example.com
- Documentation: (link to docs)

---

## License

(Specify license here)
