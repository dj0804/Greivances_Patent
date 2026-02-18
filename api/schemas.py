"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class ComplaintRequest(BaseModel):
    """Request schema for complaint urgency prediction."""
    complaint_text: str = Field(..., description="Raw complaint text")
    timestamp: Optional[datetime] = Field(None, description="Complaint timestamp")
    hostel_id: Optional[str] = Field(None, description="Hostel identifier")
    complaint_type: Optional[str] = Field(None, description="Type of complaint")
    
    class Config:
        schema_extra = {
            "example": {
                "complaint_text": "The water heater is broken and not working!!",
                "timestamp": "2026-02-17T18:30:00",
                "hostel_id": "H-101",
                "complaint_type": "maintenance"
            }
        }


class ComplaintBatchRequest(BaseModel):
    """Request schema for batch predictions."""
    complaints: List[ComplaintRequest]


class UrgencyPrediction(BaseModel):
    """Response schema for urgency prediction."""
    urgency_level: str = Field(..., description="Predicted urgency level")
    confidence: float = Field(..., description="Prediction confidence score")
    cleaned_text: str = Field(..., description="Preprocessed text")
    tokens: List[str] = Field(..., description="Extracted tokens")
    temporal_features: Optional[Dict] = Field(None, description="Temporal context")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    decision_details: Optional[Dict] = Field(None, description="Decision engine details")


class PredictionResponse(BaseModel):
    """Response wrapper for single prediction."""
    success: bool
    data: UrgencyPrediction
    processing_time_ms: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Response wrapper for batch predictions."""
    success: bool
    data: List[UrgencyPrediction]
    count: int
    processing_time_ms: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = False
    error: str
    detail: Optional[str] = None
