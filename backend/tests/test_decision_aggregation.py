"""
Comprehensive Tests for Structural Aggregation Engine
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
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


@pytest.fixture
def equal_weights_config():
    return AggregationConfig(alpha=1/3, beta=1/3, delta=1/3)


# ---------------------------------------------------------------------------
# TestMeanComputation (unittest.TestCase — no fixtures, compatible as-is)
# ---------------------------------------------------------------------------

class TestMeanComputation(unittest.TestCase):
    """Tests for mean urgency computation."""

    def setUp(self):
        self.default_config = AggregationConfig(alpha=0.4, beta=0.3, delta=0.3)

    def test_mean_computation_simple(self):
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
        complaints = [
            create_complaint("C1", 0.7),
            create_complaint("C2", 0.7),
            create_complaint("C3", 0.7),
        ]
        aggregator = StructuralAggregator(self.default_config)
        result = aggregator.aggregate(complaints)
        self.assertAlmostEqual(result["mean_urgency"], 0.7, places=6)

    def test_mean_computation_extremes(self):
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
        complaints = [
            create_complaint("C1", 0.3),
            create_complaint("C2", 0.9),
            create_complaint("C3", 0.5),
            create_complaint("C4", 0.7),
        ]
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        assert abs(result["max_urgency"] - 0.9) < 1e-6

    def test_max_at_beginning(self):
        complaints = [
            create_complaint("C1", 0.95),
            create_complaint("C2", 0.5),
            create_complaint("C3", 0.3),
        ]
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        assert abs(result["max_urgency"] - 0.95) < 1e-6

    def test_max_at_end(self):
        complaints = [
            create_complaint("C1", 0.3),
            create_complaint("C2", 0.5),
            create_complaint("C3", 0.88),
        ]
        aggregator = StructuralAggregator(AggregationConfig())
        result = aggregator.aggregate(complaints)
        assert abs(result["max_urgency"] - 0.88) < 1e-6


# ---------------------------------------------------------------------------
# Standalone test functions (lifted from plain classes)
# ---------------------------------------------------------------------------

# --- TestPercentileComputation ---

def test_percentile_90_standard():
    complaints = [create_complaint(f"C{i}", i * 0.1) for i in range(1, 11)]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert 0.85 <= result["percentile_urgency"] <= 0.95


def test_percentile_small_dataset():
    complaints = [
        create_complaint("C1", 0.2),
        create_complaint("C2", 0.8),
    ]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert 0.5 <= result["percentile_urgency"] <= 0.9


# --- TestSingleComplaint ---

def test_single_complaint_cluster():
    complaints = [create_complaint("C1", 0.75)]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert abs(result["mean_urgency"] - 0.75) < 1e-6
    assert abs(result["max_urgency"] - 0.75) < 1e-6
    assert abs(result["percentile_urgency"] - 0.75) < 1e-6
    assert result["complaint_count"] == 1


def test_single_complaint_structural_urgency(equal_weights_config):
    complaints = [create_complaint("C1", 0.6)]
    aggregator = StructuralAggregator(equal_weights_config)
    result = aggregator.aggregate(complaints)
    assert abs(result["structural_urgency"] - 0.6) < 1e-2


# --- TestIdenticalUrgency ---

def test_all_identical_urgency():
    complaints = [create_complaint(f"C{i}", 0.5) for i in range(10)]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert abs(result["mean_urgency"] - 0.5) < 1e-6
    assert abs(result["max_urgency"] - 0.5) < 1e-6
    assert abs(result["percentile_urgency"] - 0.5) < 1e-6
    assert abs(result["structural_urgency"] - 0.5) < 1e-6


def test_identical_high_urgency():
    complaints = [create_complaint(f"C{i}", 0.95) for i in range(5)]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert abs(result["structural_urgency"] - 0.95) < 1e-6


def test_identical_low_urgency():
    complaints = [create_complaint(f"C{i}", 0.1) for i in range(5)]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert abs(result["structural_urgency"] - 0.1) < 1e-6


# --- TestExtremeOutliers ---

def test_single_extreme_outlier():
    complaints = [
        create_complaint("C1", 0.1),
        create_complaint("C2", 0.1),
        create_complaint("C3", 0.1),
        create_complaint("C4", 0.1),
        create_complaint("C5", 1.0),
    ]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    expected_mean = (0.1 * 4 + 1.0) / 5
    assert abs(result["mean_urgency"] - expected_mean) < 1e-6
    assert abs(result["max_urgency"] - 1.0) < 1e-6
    assert result["structural_urgency"] > expected_mean
    assert result["structural_urgency"] < 1.0


def test_multiple_outliers():
    complaints = [
        create_complaint("C1", 0.05),
        create_complaint("C2", 0.10),
        create_complaint("C3", 0.95),
        create_complaint("C4", 0.98),
    ]
    aggregator = StructuralAggregator(AggregationConfig())
    result = aggregator.aggregate(complaints)
    assert result["percentile_urgency"] > 0.9


# --- TestWeightConfiguration ---

def test_max_dominant_weights():
    config = AggregationConfig(alpha=0.1, beta=0.8, delta=0.1)
    complaints = [
        create_complaint("C1", 0.2),
        create_complaint("C2", 0.3),
        create_complaint("C3", 0.9),
    ]
    aggregator = StructuralAggregator(config)
    result = aggregator.aggregate(complaints)
    assert result["structural_urgency"] > 0.7


def test_mean_dominant_weights():
    config = AggregationConfig(alpha=0.8, beta=0.1, delta=0.1)
    complaints = [
        create_complaint("C1", 0.2),
        create_complaint("C2", 0.3),
        create_complaint("C3", 0.9),
    ]
    aggregator = StructuralAggregator(config)
    result = aggregator.aggregate(complaints)
    mean = (0.2 + 0.3 + 0.9) / 3
    assert abs(result["structural_urgency"] - mean) < 0.15


# --- TestErrorHandling ---

def test_empty_cluster_raises_error():
    aggregator = StructuralAggregator(AggregationConfig())
    with pytest.raises(ValueError, match="empty cluster"):
        aggregator.aggregate([])


def test_invalid_urgency_score_raises_error():
    # Complaint.__post_init__ validates urgency_score in [0,1]
    with pytest.raises(ValueError):
        Complaint(
            complaint_id="C1",
            cluster_id="TEST",
            urgency_score=1.5,
            timeline_score=0.5,
            timestamp=datetime.now(),
            text="Test"
        )


def test_negative_urgency_score_raises_error():
    with pytest.raises(ValueError):
        Complaint(
            complaint_id="C1",
            cluster_id="TEST",
            urgency_score=-0.1,
            timeline_score=0.5,
            timestamp=datetime.now(),
            text="Test"
        )


# --- TestAggregateMultiple ---

def test_aggregate_multiple_clusters():
    cluster_a = [create_complaint(f"A{i}", 0.3, "CLUSTER_A") for i in range(3)]
    cluster_b = [create_complaint(f"B{i}", 0.7, "CLUSTER_B") for i in range(3)]
    cluster_complaints = {"CLUSTER_A": cluster_a, "CLUSTER_B": cluster_b}
    aggregator = StructuralAggregator(AggregationConfig())
    results = aggregator.aggregate_multiple(cluster_complaints)
    assert len(results) == 2
    assert "CLUSTER_A" in results
    assert "CLUSTER_B" in results
    assert results["CLUSTER_A"]["structural_urgency"] < results["CLUSTER_B"]["structural_urgency"]


# --- TestConvenienceFunction ---

def test_compute_structural_urgency_function():
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
