"""
Comprehensive Tests for Structural Aggregation Engine

Tests cover:
- Mean computation correctness
- Max detection correctness
- Percentile correctness
- Single complaint cluster
- Identical urgency cluster
- Extreme outlier case
- Edge cases and error handling
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from datetime import datetime
from decision_engine.engine.aggregation import StructuralAggregator, compute_structural_urgency
from decision_engine.config import AggregationConfig
from decision_engine.services.data_interface import Complaint


def create_complaint(complaint_id: str, urgency: float, cluster_id: str = "TEST") -> Complaint:
    """Helper to create test complaints."""
    return Complaint(
        complaint_id=complaint_id,
        cluster_id=cluster_id,
        urgency_score=urgency,
        timeline_score=0.5,
        timestamp=datetime.now(),
        text=f"Test complaint {complaint_id}"
    )


class TestMeanComputation(unittest.TestCase):
    """Tests for mean urgency computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.default_config = AggregationConfig(alpha=0.4, beta=0.3, delta=0.3)
    
    def test_mean_computation_simple(self):
        """Test mean computation with simple values."""
        complaints = [
            create_complaint("C1", 0.2),
            create_complaint("C2", 0.4),
            create_complaint("C3", 0.6),
        ]
        
        aggregator = StructuralAggregator(self.default_config)
        result = aggregator.aggregate(complaints)
        
        expected_mean = (0.2 + 0.4 + 0.6) / 3
        self.assertAlmostEqual(result["mean_urgency"], expected_mean, places=6)
    
    def test_mean_computation_all_same(self):
        """Test mean when all values are identical."""
        complaints = [
            create_complaint("C1", 0.7),
            create_complaint("C2", 0.7),
            create_complaint("C3", 0.7),
        ]
        
        aggregator = StructuralAggregator(self.default_config)
        result = aggregator.aggregate(complaints)
        
        self.assertAlmostEqual(result["mean_urgency"], 0.7, places=6)
    
    def test_mean_computation_extremes(self):
        """Test mean with extreme values."""
        complaints = [
            create_complaint("C1", 0.0),
            create_complaint("C2", 1.0),
        ]
        
        aggregator = StructuralAggregator(self.default_config)
        result = aggregator.aggregate(complaints)
        
        self.assertAlmostEqual(result["mean_urgency"], 0.5, places=6)


class TestMaxDetection(unittest.TestCase):
    """Tests for maximum urgency detection."""
    
    def test_max_detection_correctness(self):
        """Test that max is correctly identified."""
        complaints = [
            create_complaint("C1", 0.3),
            create_complaint("C2", 0.9),  # Maximum
            create_complaint("C3", 0.5),
            create_complaint("C4", 0.7),
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        assert abs(result["max_urgency"] - 0.9) < 1e-6
    
    def test_max_at_beginning(self):
        """Test max when it's the first element."""
        complaints = [
            create_complaint("C1", 0.95),  # Maximum
            create_complaint("C2", 0.5),
            create_complaint("C3", 0.3),
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        assert abs(result["max_urgency"] - 0.95) < 1e-6
    
    def test_max_at_end(self):
        """Test max when it's the last element."""
        complaints = [
            create_complaint("C1", 0.3),
            create_complaint("C2", 0.5),
            create_complaint("C3", 0.88),  # Maximum
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        assert abs(result["max_urgency"] - 0.88) < 1e-6


class TestPercentileComputation:
    """Tests for percentile computation."""
    
    def test_percentile_90_standard(self):
        """Test 90th percentile with standard dataset."""
        # Create 10 complaints with urgencies 0.1, 0.2, ..., 1.0
        complaints = [
            create_complaint(f"C{i}", i * 0.1)
            for i in range(1, 11)
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        # 90th percentile should be around 0.9
        assert 0.85 <= result["percentile_urgency"] <= 0.95
    
    def test_percentile_small_dataset(self):
        """Test percentile with small dataset."""
        complaints = [
            create_complaint("C1", 0.2),
            create_complaint("C2", 0.8),
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        # For 2 elements, 90th percentile interpolates
        assert 0.5 <= result["percentile_urgency"] <= 0.9


class TestSingleComplaint:
    """Tests for single-complaint clusters."""
    
    def test_single_complaint_cluster(self):
        """Test that single complaint is handled correctly."""
        complaints = [create_complaint("C1", 0.75)]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        # For single complaint, mean = max = percentile = value
        assert abs(result["mean_urgency"] - 0.75) < 1e-6
        assert abs(result["max_urgency"] - 0.75) < 1e-6
        assert abs(result["percentile_urgency"] - 0.75) < 1e-6
        assert result["complaint_count"] == 1
    
    def test_single_complaint_structural_urgency(self, equal_weights_config):
        """Test structural urgency for single complaint."""
        complaints = [create_complaint("C1", 0.6)]
        
        aggregator = StructuralAggregator(equal_weights_config)
        result = aggregator.aggregate(complaints)
        
        # With equal weights and all components = 0.6
        # structural_urgency should be 0.6
        assert abs(result["structural_urgency"] - 0.6) < 1e-2


class TestIdenticalUrgency:
    """Tests for clusters with identical urgency scores."""
    
    def test_all_identical_urgency(self):
        """Test cluster where all complaints have same urgency."""
        complaints = [create_complaint(f"C{i}", 0.5) for i in range(10)]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        # All statistics should equal the common value
        assert abs(result["mean_urgency"] - 0.5) < 1e-6
        assert abs(result["max_urgency"] - 0.5) < 1e-6
        assert abs(result["percentile_urgency"] - 0.5) < 1e-6
        assert abs(result["structural_urgency"] - 0.5) < 1e-6
    
    def test_identical_high_urgency(self):
        """Test with all high urgency."""
        complaints = [create_complaint(f"C{i}", 0.95) for i in range(5)]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        assert abs(result["structural_urgency"] - 0.95) < 1e-6
    
    def test_identical_low_urgency(self):
        """Test with all low urgency."""
        complaints = [create_complaint(f"C{i}", 0.1) for i in range(5)]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        assert abs(result["structural_urgency"] - 0.1) < 1e-6


class TestExtremeOutliers:
    """Tests for handling extreme outlier cases."""
    
    def test_single_extreme_outlier(self):
        """Test cluster with one extreme high urgency among low values."""
        complaints = [
            create_complaint("C1", 0.1),
            create_complaint("C2", 0.1),
            create_complaint("C3", 0.1),
            create_complaint("C4", 0.1),
            create_complaint("C5", 1.0),  # Extreme outlier
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        # Mean should be affected
        expected_mean = (0.1 * 4 + 1.0) / 5
        assert abs(result["mean_urgency"] - expected_mean) < 1e-6
        
        # Max should capture outlier
        assert abs(result["max_urgency"] - 1.0) < 1e-6
        
        # Structural urgency should be influenced by all components
        # With max weight of 0.3, outlier contributes but doesn't dominate
        assert result["structural_urgency"] > expected_mean
        assert result["structural_urgency"] < 1.0
    
    def test_multiple_outliers(self):
        """Test cluster with multiple outliers."""
        complaints = [
            create_complaint("C1", 0.05),
            create_complaint("C2", 0.10),
            create_complaint("C3", 0.95),  # Outlier
            create_complaint("C4", 0.98),  # Outlier
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        
        # 90th percentile should capture high outliers
        assert result["percentile_urgency"] > 0.9


class TestWeightConfiguration:
    """Tests for different weight configurations."""
    
    def test_max_dominant_weights(self):
        """Test configuration where max has dominant weight."""
        config = AggregationConfig(alpha=0.1, beta=0.8, delta=0.1)
        
        complaints = [
            create_complaint("C1", 0.2),
            create_complaint("C2", 0.3),
            create_complaint("C3", 0.9),  # High outlier
        ]
        
        aggregator = StructuralAggregator(config)
        result = aggregator.aggregate(complaints)
        
        # Structural urgency should be close to max
        assert result["structural_urgency"] > 0.7
    
    def test_mean_dominant_weights(self):
        """Test configuration where mean has dominant weight."""
        config = AggregationConfig(alpha=0.8, beta=0.1, delta=0.1)
        
        complaints = [
            create_complaint("C1", 0.2),
            create_complaint("C2", 0.3),
            create_complaint("C3", 0.9),  # High outlier
        ]
        
        aggregator = StructuralAggregator(config)
        result = aggregator.aggregate(complaints)
        
        # Structural urgency should be close to mean
        mean = (0.2 + 0.3 + 0.9) / 3
        assert abs(result["structural_urgency"] - mean) < 0.15


class TestErrorHandling:
    """Tests for error handling and validation."""
    
    def test_empty_cluster_raises_error(self):
        """Test that empty cluster raises ValueError."""
        aggregator = StructuralAggregator(AggregationConfig())
        
        with pytest.raises(ValueError, match="empty cluster"):
            aggregator.aggregate([])
    
    def test_invalid_urgency_score_raises_error(self):
        """Test that invalid urgency scores raise ValueError."""
        # Invalid score > 1.0
        complaints = [
            Complaint(
                complaint_id="C1",
                cluster_id="TEST",
                urgency_score=1.5,  # Invalid
                timeline_score=0.5,
                timestamp=datetime.now(),
                text="Test"
            )
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        
        with pytest.raises(ValueError, match="Invalid urgency scores"):
            aggregator.aggregate(complaints)
    
    def test_negative_urgency_score_raises_error(self):
        """Test that negative urgency scores raise ValueError."""
        complaints = [
            Complaint(
                complaint_id="C1",
                cluster_id="TEST",
                urgency_score=-0.1,  # Invalid
                timeline_score=0.5,
                timestamp=datetime.now(),
                text="Test"
            )
        ]
        
        aggregator = StructuralAggregator(AggregationConfig())
        
        with pytest.raises(ValueError, match="Invalid urgency scores"):
            aggregator.aggregate(complaints)


class TestAggregateMultiple:
    """Tests for batch aggregation."""
    
    def test_aggregate_multiple_clusters(self):
        """Test aggregating multiple clusters at once."""
        cluster_a = [create_complaint(f"A{i}", 0.3, "CLUSTER_A") for i in range(3)]
        cluster_b = [create_complaint(f"B{i}", 0.7, "CLUSTER_B") for i in range(3)]
        
        cluster_complaints = {
            "CLUSTER_A": cluster_a,
            "CLUSTER_B": cluster_b
        }
        
        aggregator = StructuralAggregator(AggregationConfig())
        results = aggregator.aggregate_multiple(cluster_complaints)
        
        assert len(results) == 2
        assert "CLUSTER_A" in results
        assert "CLUSTER_B" in results
        assert results["CLUSTER_A"]["structural_urgency"] < results["CLUSTER_B"]["structural_urgency"]


class TestConvenienceFunction:
    """Tests for the convenience function."""
    
    def test_compute_structural_urgency_function(self):
        """Test the convenience function."""
        complaints = [
            create_complaint("C1", 0.5),
            create_complaint("C2", 0.6),
            create_complaint("C3", 0.7),
        ]
        
        urgency = compute_structural_urgency(complaints)
        
        assert 0.0 <= urgency <= 1.0
        assert isinstance(urgency, float)


if __name__ == "__main__":
    unittest.main()
