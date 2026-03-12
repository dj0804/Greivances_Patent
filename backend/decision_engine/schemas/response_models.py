"""
Pydantic models for API responses.

These models define the contract for data returned by the Decision Engine.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ClusterUrgencyBreakdown(BaseModel):
    """Detailed breakdown of urgency computation for a cluster."""
    
    structural_urgency: float = Field(..., description="Aggregated structural urgency component")
    temporal_urgency: float = Field(..., description="Temporal dynamics component")
    raw_urgency: float = Field(..., description="Fused urgency before calibration")
    size_normalized_urgency: float = Field(..., description="After size penalty")
    final_urgency: float = Field(..., description="Final calibrated urgency score")
    
    # Component details
    mean_urgency: float = Field(..., description="Mean of complaint urgencies")
    max_urgency: float = Field(..., description="Maximum complaint urgency")
    percentile_90_urgency: float = Field(..., description="90th percentile urgency")
    
    # Temporal details
    volume_ratio: float = Field(..., description="Recent vs historical volume ratio")
    arrival_rate: float = Field(..., description="Complaint arrival rate (per hour)")
    
    # Cluster metadata
    complaint_count: int = Field(..., description="Number of complaints in cluster")
    previous_urgency: Optional[float] = Field(None, description="Previous urgency (for smoothing)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "structural_urgency": 0.75,
                "temporal_urgency": 0.65,
                "raw_urgency": 0.71,
                "size_normalized_urgency": 0.68,
                "final_urgency": 0.69,
                "mean_urgency": 0.70,
                "max_urgency": 0.85,
                "percentile_90_urgency": 0.80,
                "volume_ratio": 1.5,
                "arrival_rate": 2.3,
                "complaint_count": 15,
                "previous_urgency": 0.70
            }
        }


class ClusterSummary(BaseModel):
    """Summary of a complaint cluster."""
    
    cluster_id: str = Field(..., description="Cluster identifier")
    final_urgency: float = Field(..., ge=0.0, le=1.0, description="Final urgency score")
    
    complaint_count: int = Field(..., description="Number of complaints in cluster")
    
    top_complaints: List[str] = Field(
        ..., 
        description="IDs of top-k highest urgency complaints"
    )
    
    summary_text: str = Field(..., description="Concatenated summary from top complaints")
    
    breakdown: ClusterUrgencyBreakdown = Field(..., description="Detailed urgency breakdown")
    
    earliest_complaint: datetime = Field(..., description="Timestamp of earliest complaint")
    latest_complaint: datetime = Field(..., description="Timestamp of latest complaint")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class ProcessDecisionResponse(BaseModel):
    """Response from processing complaints."""
    
    status: str = Field(..., description="Processing status")
    clusters_processed: int = Field(..., description="Number of clusters processed")
    total_complaints: int = Field(..., description="Total complaints processed")
    
    ranked_clusters: List[ClusterSummary] = Field(
        ..., 
        description="Clusters ranked by final urgency (descending)"
    )
    
    processing_timestamp: datetime = Field(..., description="When processing completed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "clusters_processed": 15,
                "total_complaints": 234,
                "ranked_clusters": [],
                "processing_timestamp": "2024-02-15T17:00:00Z"
            }
        }


class GetRankedResponse(BaseModel):
    """Response for retrieving ranked clusters."""
    
    clusters: List[ClusterSummary] = Field(
        ..., 
        description="Ranked clusters"
    )
    
    total_clusters: int = Field(..., description="Total number of clusters available")
    
    retrieved_at: datetime = Field(..., description="Timestamp of retrieval")
    
    class Config:
        json_schema_extra = {
            "example": {
                "clusters": [],
                "total_clusters": 15,
                "retrieved_at": "2024-02-15T17:05:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid urgency score provided",
                "details": {"field": "urgency_score", "value": 1.5}
            }
        }
