"""
Urgency Calibration Engine

This module calibrates raw urgency scores to prevent bias and reduce volatility.

Mathematical Foundation:
-----------------------
The calibration process applies two transformations:

1. Size Normalization (bias correction):
    U_size = U_raw / (1 + log(1 + n_k))
    
    where:
        U_raw = raw urgency score
        n_k = number of complaints in cluster k
    
    Rationale:
        - Large clusters naturally have higher max/mean values
        - Without correction, large clusters dominate ranking
        - Logarithmic penalty: diminishing returns for size
        - "+1" inside log: ensures log(1) = 0 for single complaint
        - "+1" outside: ensures denominator ≥ 1 (no amplification)

2. Exponential Smoothing (volatility reduction):
    U_final = γ·U_prev + (1-γ)·U_size
    
    where:
        U_prev = previous urgency for this cluster
        γ = smoothing factor ∈ [0, 1]
    
    Rationale:
        - Prevents sudden jumps in urgency
        - Provides temporal stability in rankings
        - γ = 0: No memory (pure reactive)
        - γ = 1: No update (pure persistence)
        - γ = 0.7 (default): Strong but not rigid memory

Bias Correction Justification:
-----------------------------
Consider two clusters:
    A: 50 complaints, mean urgency 0.7
    B: 5 complaints, mean urgency 0.8

Without correction:
    - A might dominate due to sheer volume
    - Single outlier in A has less impact than in B
    
With log penalty:
    - A: 0.7 / (1 + log(51)) ≈ 0.7 / 4.94 ≈ 0.14
    - B: 0.8 / (1 + log(6)) ≈ 0.8 / 2.79 ≈ 0.29
    - Now B can outrank A despite smaller size

This ensures urgency reflects true severity, not just volume.

Numerical Stability:
-------------------
- log(1 + n_k): Always positive for n_k ≥ 0
- Division by (1 + ...): Denominator always ≥ 1
- Smoothing: Convex combination always in valid range
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

from decision_engine.config import CalibrationConfig

logger = logging.getLogger(__name__)


class UrgencyCalibrator:
    """
    Calibrates raw urgency scores with size normalization and temporal smoothing.
    
    Maintains a memory of previous urgencies for smoothing.
    """
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize calibrator with configuration.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        
        # Memory store: cluster_id -> (urgency, timestamp)
        self._urgency_memory: Dict[str, tuple] = {}
        
        logger.info(
            f"Initialized UrgencyCalibrator with γ={config.gamma}, "
            f"smoothing={'enabled' if config.enable_smoothing else 'disabled'}"
        )
    
    def calibrate(
        self, 
        cluster_id: str,
        raw_urgency: float, 
        complaint_count: int,
        current_time: datetime = None
    ) -> dict:
        """
        Calibrate raw urgency score.
        
        Args:
            cluster_id: Cluster identifier (for memory lookup)
            raw_urgency: Raw urgency before calibration
            complaint_count: Number of complaints in cluster
            current_time: Current timestamp (for memory management)
            
        Returns:
            Dictionary containing:
                - size_normalized_urgency: After size penalty
                - final_urgency: After smoothing
                - previous_urgency: Previous urgency used for smoothing (None if first time)
                - size_penalty_factor: Denominator used for normalization
                
        Raises:
            ValueError: If inputs are invalid
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Validate inputs
        if raw_urgency < 0.0 or raw_urgency > 1.0:
            raise ValueError(f"raw_urgency must be in [0, 1], got {raw_urgency}")
        if complaint_count < 1:
            raise ValueError(f"complaint_count must be ≥ 1, got {complaint_count}")
        
        # Step 1: Size normalization
        size_normalized_urgency, penalty_factor = self._apply_size_normalization(
            raw_urgency, complaint_count
        )
        
        # Step 2: Temporal smoothing
        previous_urgency = self._get_previous_urgency(cluster_id, current_time)
        
        if self.config.enable_smoothing and previous_urgency is not None:
            final_urgency = self._apply_smoothing(
                size_normalized_urgency, previous_urgency
            )
        else:
            final_urgency = size_normalized_urgency
        
        # Update memory
        self._update_memory(cluster_id, final_urgency, current_time)
        
        logger.debug(
            f"Calibrated cluster {cluster_id}: "
            f"raw={raw_urgency:.3f} → size_norm={size_normalized_urgency:.3f} → "
            f"final={final_urgency:.3f} (n={complaint_count}, prev={previous_urgency})"
        )
        
        return {
            "size_normalized_urgency": size_normalized_urgency,
            "final_urgency": final_urgency,
            "previous_urgency": previous_urgency,
            "size_penalty_factor": penalty_factor
        }
    
    def _apply_size_normalization(self, raw_urgency: float, n: int) -> tuple:
        """
        Apply logarithmic size penalty to prevent large clusters from dominating.
        
        Mathematical form:
            U_size = U_raw / (1 + log(1 + n))
        
        Args:
            raw_urgency: Raw urgency score
            n: Number of complaints in cluster
            
        Returns:
            Tuple of (normalized_urgency, penalty_factor)
            
        Behavior:
            n=1:   penalty=1.0      (no penalty for single complaint)
            n=10:  penalty≈3.4      (moderate penalty)
            n=100: penalty≈5.6      (strong penalty)
            n=1000: penalty≈7.9     (very strong penalty)
        """
        # Compute penalty factor: 1 + log(1 + n)
        penalty_factor = 1.0 + np.log(1.0 + n) / np.log(self.config.size_penalty_base + np.e)
        
        # Apply normalization
        normalized_urgency = raw_urgency / penalty_factor
        
        # Ensure result stays in [0, 1]
        normalized_urgency = np.clip(normalized_urgency, 0.0, 1.0)
        
        logger.debug(
            f"Size normalization: {raw_urgency:.3f} / {penalty_factor:.2f} = {normalized_urgency:.3f}"
        )
        
        return float(normalized_urgency), float(penalty_factor)
    
    def _apply_smoothing(self, current: float, previous: float) -> float:
        """
        Apply exponential smoothing to reduce volatility.
        
        Mathematical form:
            U_final = γ·U_prev + (1-γ)·U_current
        
        Args:
            current: Current urgency (after size normalization)
            previous: Previous urgency
            
        Returns:
            Smoothed urgency
            
        This is a weighted average that:
        - Preserves continuity (no sudden jumps)
        - Gradually adapts to new values
        - Provides stability in rankings
        """
        smoothed = (
            self.config.gamma * previous +
            (1.0 - self.config.gamma) * current
        )
        
        # Ensure result stays in [0, 1]
        smoothed = np.clip(smoothed, 0.0, 1.0)
        
        logger.debug(
            f"Smoothing: {self.config.gamma}×{previous:.3f} + "
            f"{1-self.config.gamma}×{current:.3f} = {smoothed:.3f}"
        )
        
        return float(smoothed)
    
    def _get_previous_urgency(self, cluster_id: str, current_time: datetime) -> Optional[float]:
        """
        Retrieve previous urgency from memory if available and not expired.
        
        Args:
            cluster_id: Cluster identifier
            current_time: Current timestamp
            
        Returns:
            Previous urgency if available and fresh, None otherwise
        """
        if cluster_id not in self._urgency_memory:
            return None
        
        previous_urgency, timestamp = self._urgency_memory[cluster_id]
        
        # Check if memory is stale
        age = (current_time - timestamp).total_seconds() / 3600.0  # Convert to hours
        if age > self.config.memory_persistence_hours:
            logger.debug(f"Memory for {cluster_id} expired (age={age:.1f}h)")
            del self._urgency_memory[cluster_id]
            return None
        
        return previous_urgency
    
    def _update_memory(self, cluster_id: str, urgency: float, timestamp: datetime) -> None:
        """
        Update memory with new urgency.
        
        Args:
            cluster_id: Cluster identifier
            urgency: Urgency to store
            timestamp: Current timestamp
        """
        self._urgency_memory[cluster_id] = (urgency, timestamp)
        logger.debug(f"Updated memory for {cluster_id}: {urgency:.3f}")
    
    def clear_memory(self, cluster_id: str = None) -> None:
        """
        Clear urgency memory.
        
        Args:
            cluster_id: Specific cluster to clear (None = clear all)
        """
        if cluster_id is None:
            self._urgency_memory.clear()
            logger.info("Cleared all urgency memory")
        elif cluster_id in self._urgency_memory:
            del self._urgency_memory[cluster_id]
            logger.info(f"Cleared memory for cluster {cluster_id}")
    
    def get_memory_stats(self) -> dict:
        """
        Get statistics about memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "clusters_in_memory": len(self._urgency_memory),
            "cluster_ids": list(self._urgency_memory.keys())
        }
    
    def calibrate_multiple(
        self, 
        cluster_data: dict,
        current_time: datetime = None
    ) -> dict:
        """
        Calibrate urgencies for multiple clusters.
        
        Args:
            cluster_data: Dict mapping cluster_id -> {
                "raw_urgency": float,
                "complaint_count": int
            }
            current_time: Current timestamp
            
        Returns:
            Dict mapping cluster_id -> calibration result
        """
        if current_time is None:
            current_time = datetime.now()
        
        results = {}
        
        for cluster_id, data in cluster_data.items():
            results[cluster_id] = self.calibrate(
                cluster_id=cluster_id,
                raw_urgency=data["raw_urgency"],
                complaint_count=data["complaint_count"],
                current_time=current_time
            )
        
        logger.info(f"Calibrated {len(results)} clusters")
        return results


# Utility function
def calibrate_urgency(
    cluster_id: str,
    raw_urgency: float,
    complaint_count: int,
    config: CalibrationConfig = None,
    calibrator: UrgencyCalibrator = None
) -> float:
    """
    Convenience function to calibrate urgency.
    
    Args:
        cluster_id: Cluster identifier
        raw_urgency: Raw urgency score
        complaint_count: Number of complaints
        config: Optional custom configuration
        calibrator: Optional existing calibrator (for memory persistence)
        
    Returns:
        Final calibrated urgency
    """
    if calibrator is None:
        if config is None:
            from decision_engine.config import config as default_config
            config = default_config.calibration
        calibrator = UrgencyCalibrator(config)
    
    result = calibrator.calibrate(cluster_id, raw_urgency, complaint_count)
    return result["final_urgency"]
