"""
Pydantic models for API requests.

These models define the contract for incoming data to the Decision Engine.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class ComplaintInput(BaseModel):
    """Input model for a single complaint.
    
    This is the minimal interface required by the Decision Engine.
    Clustering-source agnostic - works with any clustering mechanism.
    """
    complaint_id: str = Field(..., description="Unique identifier for the complaint")
    cluster_id: str = Field(..., description="Cluster assignment (from any clustering source)")
    urgency_score: float = Field(..., ge=0.0, le=1.0, description="Complaint-level urgency score")
    timeline_score: float = Field(..., ge=0.0, le=1.0, description="Timeline-based urgency factor")
    timestamp: datetime = Field(..., description="Complaint submission timestamp")
    text: str = Field(..., min_length=1, description="Complaint text content")
    
    @validator('urgency_score', 'timeline_score')
    def validate_scores(cls, v):
        """Ensure scores are within [0, 1] range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Score must be in [0, 1], got {v}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "complaint_id": "CMP-2024-001234",
                "cluster_id": "CLUSTER-42",
                "urgency_score": 0.85,
                "timeline_score": 0.75,
                "timestamp": "2024-02-15T14:30:00Z",
                "text": "Critical system outage affecting multiple users"
            }
        }


class ProcessDecisionRequest(BaseModel):
    """Request to process complaints and compute cluster urgencies.
    
    The engine will group complaints by cluster_id and compute
    aggregated urgency for each cluster.
    """
    complaints: List[ComplaintInput] = Field(
        ..., 
        min_items=1,
        description="List of complaints to process"
    )
    
    recompute_all: bool = Field(
        default=True,
        description="Whether to recompute all clusters or only new ones"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "complaints": [
                    {
                        "complaint_id": "CMP-001",
                        "cluster_id": "CLUSTER-1",
                        "urgency_score": 0.8,
                        "timeline_score": 0.7,
                        "timestamp": "2024-02-15T14:30:00Z",
                        "text": "System down"
                    }
                ],
                "recompute_all": True
            }
        }


class GetRankedRequest(BaseModel):
    """Request parameters for retrieving ranked clusters."""
    
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of top clusters to return (None = all)"
    )
    
    min_urgency_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum urgency threshold for filtering"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "top_n": 10,
                "min_urgency_threshold": 0.5
            }
        }
