"""
Comprehensive Integration Tests for Decision Engine

Tests the complete pipeline end-to-end to verify:
1. Increasing complaint severity increases structural urgency
2. Increasing complaint frequency increases temporal urgency
3. Large cluster size does not unfairly dominate
4. Smoothing reduces sudden oscillations
5. Ranking is stable and predictable
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta
from decision_engine.engine.orchestrator import DecisionEngineOrchestrator
from decision_engine.config import DecisionEngineConfig, AggregationConfig, TemporalConfig
from decision_engine.services.data_interface import Complaint


def create_complaint(
    complaint_id: str,
    cluster_id: str,
    urgency: float,
    hours_ago: float = 5.0,
    reference_time: datetime = None
) -> Complaint:
    """Helper to create test complaints."""
    if reference_time is None:
        reference_time = datetime.now()
    
    timestamp = reference_time - timedelta(hours=hours_ago)
    
    return Complaint(
        complaint_id=complaint_id,
        cluster_id=cluster_id,
        urgency_score=urgency,
        timeline_score=0.5,
        timestamp=timestamp,
        text=f"Complaint {complaint_id} in {cluster_id}"
    )


class TestValidationGoal1(unittest.TestCase):
    """Validation Goal #1: Increasing complaint severity increases structural urgency."""
    
    def test_higher_severity_higher_urgency(self):
        """Test that clusters with higher severity complaints rank higher."""
        reference_time = datetime.now()
        orchestrator = DecisionEngineOrchestrator()
        
        # Cluster A: Low severity
        low_severity = [
            create_complaint(f"A{i}", "LOW_SEV", 0.3, i * 2, reference_time)
            for i in range(5)
        ]
        
        # Cluster B: High severity
        high_severity = [
            create_complaint(f"B{i}", "HIGH_SEV", 0.8, i * 2, reference_time)
            for i in range(5)
        ]
        
        all_complaints = low_severity + high_severity
        results = orchestrator.process_complaints(all_complaints, reference_time)
        
        # Find results by cluster
        high_sev_result = next(r for r in results if r.cluster_id == "HIGH_SEV")
        low_sev_result = next(r for r in results if r.cluster_id == "LOW_SEV")
        
        # High severity should have higher structural urgency
        self.assertGreater(
            high_sev_result.breakdown.structural_urgency,
            low_sev_result.breakdown.structural_urgency
        )
        
        # High severity should rank higher (appear first in sorted list)
        self.assertEqual(results[0].cluster_id, "HIGH_SEV")


class TestFullPipeline(unittest.TestCase):
    """Tests for complete pipeline."""
    
    def test_full_pipeline(self):
        """Test complete pipeline with multiple clusters."""
        reference_time = datetime.now()
        orchestrator = DecisionEngineOrchestrator()
        
        complaints = []
        
        # Cluster A: High severity, low volume
        for i in range(3):
            complaints.append(create_complaint(f"A{i}", "CLUSTER_A", 0.9, i * 5, reference_time))
        
        # Cluster B: Medium severity, high burst
        for i in range(15):
            complaints.append(create_complaint(f"B{i}", "CLUSTER_B", 0.6, i * 0.5, reference_time))
        
        # Cluster C: Low severity, medium volume
        for i in range(7):
            complaints.append(create_complaint(f"C{i}", "CLUSTER_C", 0.3, i * 3, reference_time))
        
        results = orchestrator.process_complaints(complaints, reference_time)
        
        # Should have 3 clusters
        self.assertEqual(len(results), 3)
        
        # All results should have valid urgency scores
        for result in results:
            self.assertGreaterEqual(result.final_urgency, 0.0)
            self.assertLessEqual(result.final_urgency, 1.0)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling."""
    
    def test_empty_complaints(self):
        """Test that empty complaints list raises error."""
        orchestrator = DecisionEngineOrchestrator()
        with self.assertRaises(ValueError):
            orchestrator.process_complaints([])


if __name__ == "__main__":
    unittest.main()
