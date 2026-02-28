"""
Temporal and contextual feature encoder.
Converts timestamps and metadata into JSON-structured features.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional


class TemporalEncoder:
    """
    Encodes temporal and contextual information into structured features.
    """
    
    def __init__(self):
        # Define peak complaint hours (evenings/nights)
        self.peak_hours = range(18, 24)
        # Weekend days
        self.weekend_days = [5, 6]  # Saturday, Sunday
    
    def encode(
        self,
        timestamp: datetime,
        hostel_id: Optional[str] = None,
        complaint_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode temporal and contextual features as JSON.
        
        Args:
            timestamp: When the complaint was filed
            hostel_id: Hostel identifier
            complaint_type: Type of complaint (maintenance, food, etc.)
            **kwargs: Additional metadata
            
        Returns:
            Dictionary of encoded features
        """
        features = {
            "temporal": self._encode_temporal(timestamp),
            "context": {
                "hostel_id": hostel_id,
                "complaint_type": complaint_type,
            },
            "metadata": kwargs
        }
        
        return features
    
    def _encode_temporal(self, timestamp: datetime) -> Dict[str, Any]:
        """Extract temporal features from timestamp."""
        return {
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_peak_hour": timestamp.hour in self.peak_hours,
            "is_weekend": timestamp.weekday() in self.weekend_days,
            "is_night": 0 <= timestamp.hour < 6,
            "month": timestamp.month,
            "day_of_month": timestamp.day,
        }
    
    def to_json(self, features: Dict[str, Any]) -> str:
        """Convert features to JSON string."""
        return json.dumps(features, indent=2)
    
    def from_json(self, json_str: str) -> Dict[str, Any]:
        """Load features from JSON string."""
        return json.loads(json_str)


def encode_temporal_features(
    timestamp: datetime,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for temporal encoding.
    
    Args:
        timestamp: Complaint timestamp
        **kwargs: Additional metadata
        
    Returns:
        Encoded features dictionary
    """
    encoder = TemporalEncoder()
    return encoder.encode(timestamp, **kwargs)
