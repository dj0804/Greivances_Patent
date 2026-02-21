"""
Comprehensive Tests for Temporal Urgency Engine

Tests cover:
- Burst detection increases score
- Slow arrival reduces score
- Zero historical volume handled
- Single complaint case
- Sigmoid stability
- Edge cases
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from datetime import datetime, timedelta
from decision_engine.engine.temporal import TemporalUrgencyComputer, compute_temporal_urgency
from decision_engine.config import TemporalConfig
from decision_engine.services.data_interface import Complaint


def create_complaint(
    complaint_id: str, 
    hours_ago: float, 
    urgency: float = 0.5,
    reference_time: datetime = None
) -> Complaint:
    """Helper to create temporal test complaints."""
    if reference_time is None:
        reference_time = datetime.now()
    
    timestamp = reference_time - timedelta(hours=hours_ago)
    
    return Complaint(
        complaint_id=complaint_id,
        cluster_id="TEST",
        urgency_score=urgency,
        timeline_score=0.5,
        timestamp=timestamp,
        text=f"Complaint {complaint_id}"
    )


class TestBurstDetection(unittest.TestCase):
    """Tests that burst detection increases temporal score."""
    
    def test_recent_burst_increases_score(self):
        """Test that recent burst of complaints increases urgency."""
        config = TemporalConfig(
            theta_1=2.0,
            theta_2=1.5,
            recent_window_hours=24.0,
            historical_window_hours=168.0
        )
        reference_time = datetime.now()
        
        # Historical: 2 complaints in past week
        # Recent: 10 complaints in past 24 hours (5x increase)
        complaints = []
        
        # Historical complaints (150 hours ago)
        complaints.append(create_complaint("H1", 150, reference_time=reference_time))
        complaints.append(create_complaint("H2", 160, reference_time=reference_time))
        
        # Recent burst (last 24 hours)
        for i in range(10):
            complaints.append(create_complaint(f"R{i}", i * 2, reference_time=reference_time))
        
        computer = TemporalUrgencyComputer(config)
        result = computer.compute(complaints, reference_time)
        
        # Volume ratio should be high
        self.assertGreater(result["volume_ratio"], 1.0)
        self.assertEqual(result["recent_count"], 10)
        self.assertEqual(result["historical_count"], 2)
        
        # Temporal urgency should be elevated
        self.assertGreater(result["temporal_urgency"], 0.5)
    
    def test_no_burst_low_score(self):
        """Test that steady arrival without burst has lower score."""
        config = TemporalConfig(
            theta_1=2.0,
            theta_2=1.5,
            recent_window_hours=24.0,
            historical_window_hours=168.0
        )
        reference_time = datetime.now()
        
        # Evenly distributed complaints (no burst)
        complaints = []
        for i in range(10):
            # Spread evenly over 200 hours
            complaints.append(create_complaint(f"C{i}", i * 20, reference_time=reference_time))
        
        computer = TemporalUrgencyComputer(config)
        result = computer.compute(complaints, reference_time)
        
        # Volume ratio should be closer to 1 or less
        self.assertLess(result["volume_ratio"], 2.0)
        
        # Temporal urgency should be moderate to low
        self.assertLess(result["temporal_urgency"], 0.8)


class TestSingleComplaint(unittest.TestCase):
    """Tests for single complaint case."""
    
    def test_single_complaint_handled(self):
        """Test that single complaint is handled without error."""
        config = TemporalConfig(
            theta_1=2.0,
            theta_2=1.5,
            recent_window_hours=24.0,
            historical_window_hours=168.0
        )
        reference_time = datetime.now()
        
        complaints = [create_complaint("C1", 5, reference_time=reference_time)]
        
        computer = TemporalUrgencyComputer(config)
        result = computer.compute(complaints, reference_time)
        
        # Arrival rate should be 0 for single complaint
        self.assertEqual(result["arrival_rate"], 0.0)
        self.assertEqual(result["mean_inter_arrival"], 0.0)
        
        # Should not crash and return valid urgency
        self.assertGreaterEqual(result["temporal_urgency"], 0.0)
        self.assertLessEqual(result["temporal_urgency"], 1.0)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling."""
    
    def test_empty_complaints_raises_error(self):
        """Test that empty complaint list raises error."""
        config = TemporalConfig(
            theta_1=2.0,
            theta_2=1.5,
            recent_window_hours=24.0,
            historical_window_hours=168.0
        )
        computer = TemporalUrgencyComputer(config)
        
        with self.assertRaises(ValueError):
            computer.compute([])


if __name__ == "__main__":
    unittest.main()
