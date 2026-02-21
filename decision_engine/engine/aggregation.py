"""
Structural Urgency Aggregation Engine

This module computes cluster-level structural urgency by aggregating
complaint-level urgency scores using statistical measures.

Mathematical Foundation:
-----------------------
Given a cluster k with complaints having urgency scores U₁, U₂, ..., Uₙ:

    U_struct = α·μ_k + β·M_k + δ·P_k

where:
    μ_k = mean(U₁, U₂, ..., Uₙ)     - Average urgency severity
    M_k = max(U₁, U₂, ..., Uₙ)      - Worst-case severity
    P_k = percentile_90(U₁, ..., Uₙ) - High-severity prevalence
    
    α + β + δ = 1.0 (interpretability constraint)

Rationale:
----------
- Mean (μ_k): Represents overall cluster severity
- Max (M_k): Ensures critical outliers aren't lost in averaging
- 90th percentile (P_k): Robust measure of high-severity prevalence,
                         less sensitive to single outliers than max

This combination balances between:
    - Overall severity (mean)
    - Worst-case attention (max)
    - Robust high-severity detection (percentile)
"""

import numpy as np
from typing import List
import logging

from decision_engine.config import AggregationConfig
from decision_engine.services.data_interface import Complaint

logger = logging.getLogger(__name__)


class StructuralAggregator:
    """
    Aggregates complaint-level urgencies into cluster-level structural urgency.
    
    This class is stateless and thread-safe - all computation is done per call.
    """
    
    def __init__(self, config: AggregationConfig):
        """
        Initialize aggregator with configuration.
        
        Args:
            config: Aggregation configuration containing weights
        """
        self.config = config
        logger.info(
            f"Initialized StructuralAggregator with α={config.alpha}, "
            f"β={config.beta}, δ={config.delta}"
        )
    
    def aggregate(self, complaints: List[Complaint]) -> dict:
        """
        Compute structural urgency for a cluster.
        
        Args:
            complaints: List of complaints in the cluster
            
        Returns:
            Dictionary containing:
                - structural_urgency: Final aggregated score
                - mean_urgency: Mean component
                - max_urgency: Max component
                - percentile_urgency: Percentile component
                - complaint_count: Number of complaints
                
        Raises:
            ValueError: If complaints list is empty or scores are invalid
        """
        if not complaints:
            raise ValueError("Cannot aggregate empty cluster")
        
        # Extract and validate urgency scores
        urgency_scores = self._extract_urgency_scores(complaints)
        
        # Compute statistical components
        mean_urgency = self._compute_mean(urgency_scores)
        max_urgency = self._compute_max(urgency_scores)
        percentile_urgency = self._compute_percentile(
            urgency_scores, 
            self.config.percentile_threshold
        )
        
        # Compute weighted structural urgency
        structural_urgency = (
            self.config.alpha * mean_urgency +
            self.config.beta * max_urgency +
            self.config.delta * percentile_urgency
        )
        
        logger.debug(
            f"Aggregated {len(complaints)} complaints: "
            f"μ={mean_urgency:.3f}, M={max_urgency:.3f}, "
            f"P90={percentile_urgency:.3f} → U_struct={structural_urgency:.3f}"
        )
        
        return {
            "structural_urgency": structural_urgency,
            "mean_urgency": mean_urgency,
            "max_urgency": max_urgency,
            "percentile_urgency": percentile_urgency,
            "complaint_count": len(complaints)
        }
    
    def _extract_urgency_scores(self, complaints: List[Complaint]) -> np.ndarray:
        """
        Extract urgency scores from complaints and validate.
        
        Args:
            complaints: List of complaints
            
        Returns:
            NumPy array of urgency scores
            
        Raises:
            ValueError: If any score is out of bounds [0, 1]
        """
        scores = np.array([c.urgency_score for c in complaints])
        
        # Validate all scores are in valid range
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            invalid = scores[(scores < 0.0) | (scores > 1.0)]
            raise ValueError(
                f"Invalid urgency scores found (must be in [0, 1]): {invalid}"
            )
        
        return scores
    
    def _compute_mean(self, scores: np.ndarray) -> float:
        """
        Compute mean urgency score.
        
        Mathematical form: μ = (1/n) Σᵢ Uᵢ
        
        Args:
            scores: Array of urgency scores
            
        Returns:
            Mean urgency score
        """
        return float(np.mean(scores))
    
    def _compute_max(self, scores: np.ndarray) -> float:
        """
        Compute maximum urgency score.
        
        Mathematical form: M = max(U₁, U₂, ..., Uₙ)
        
        Args:
            scores: Array of urgency scores
            
        Returns:
            Maximum urgency score
        """
        return float(np.max(scores))
    
    def _compute_percentile(self, scores: np.ndarray, percentile: float) -> float:
        """
        Compute percentile of urgency scores.
        
        Uses linear interpolation for non-integer ranks (NumPy default).
        
        Args:
            scores: Array of urgency scores
            percentile: Percentile to compute (0-100)
            
        Returns:
            Percentile value
            
        Special cases:
            - Single complaint: Returns that complaint's score
            - Two complaints: Linearly interpolates
        """
        if len(scores) == 1:
            # For single complaint, percentile equals the score
            return float(scores[0])
        
        return float(np.percentile(scores, percentile))
    
    def aggregate_multiple(self, cluster_complaints: dict) -> dict:
        """
        Aggregate urgency for multiple clusters at once.
        
        Args:
            cluster_complaints: Dict mapping cluster_id -> List[Complaint]
            
        Returns:
            Dict mapping cluster_id -> aggregation result
            
        Raises:
            ValueError: If any cluster has invalid data
        """
        results = {}
        
        for cluster_id, complaints in cluster_complaints.items():
            try:
                results[cluster_id] = self.aggregate(complaints)
                logger.debug(f"Aggregated cluster {cluster_id}")
            except ValueError as e:
                logger.error(f"Failed to aggregate cluster {cluster_id}: {e}")
                raise ValueError(f"Cluster {cluster_id}: {e}") from e
        
        logger.info(f"Aggregated {len(results)} clusters")
        return results


# Utility functions for one-off aggregation
def compute_structural_urgency(
    complaints: List[Complaint],
    config: AggregationConfig = None
) -> float:
    """
    Convenience function to compute structural urgency for a cluster.
    
    Args:
        complaints: List of complaints in the cluster
        config: Optional custom configuration (uses default if None)
        
    Returns:
        Structural urgency score
    """
    if config is None:
        from decision_engine.config import config as default_config
        config = default_config.aggregation
    
    aggregator = StructuralAggregator(config)
    result = aggregator.aggregate(complaints)
    return result["structural_urgency"]
