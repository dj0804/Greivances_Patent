"""
Temporal Urgency Computation Engine

This module computes temporal urgency based on complaint arrival patterns,
volume trends, and burst detection.

Mathematical Foundation:
-----------------------
Given a cluster k with complaints at timestamps t₁, t₂, ..., tₙ:

1. Volume Ratio (burst detection):
    R_k = V_recent / (V_historical + ε)
    
    where:
        V_recent = count of complaints in recent window
        V_historical = count of complaints in historical window
        ε = small smoothing constant to prevent division by zero

2. Arrival Rate (complaint frequency):
    λ_k = 1 / mean_inter_arrival_time
         = (n-1) / (t_n - t_1)  for n > 1
    
    Measured in complaints per hour.

3. Temporal Intensity (sigmoid transformation):
    T_k = σ(θ₁·R_k + θ₂·λ_k)
    
    where σ(x) = 1 / (1 + e^(-x)) is the sigmoid function
    
    Sigmoid provides:
        - Smoothness: Continuous differentiable mapping
        - Boundedness: Output always in (0, 1)
        - Interpretability: Maps intensity to probability-like score

Rationale:
----------
- Volume ratio (R_k): Detects sudden bursts or escalations
  High R_k → Recent surge in complaints
  
- Arrival rate (λ_k): Captures complaint frequency
  High λ_k → Rapid complaint submission
  
- Sigmoid: Nonlinear transformation that:
  1. Prevents extreme values from dominating
  2. Provides smooth transitions
  3. Bounds output to [0, 1] range

Numerical Stability:
-------------------
- Division by zero: Protected by ε
- Sigmoid overflow: Clip inputs to prevent e^x overflow
- Single complaint: Handled as special case
- Out-of-order timestamps: Sorted before processing
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import logging

from decision_engine.config import TemporalConfig
from decision_engine.services.data_interface import Complaint

logger = logging.getLogger(__name__)


class TemporalUrgencyComputer:
    """
    Computes temporal urgency based on complaint arrival patterns.
    
    This class is stateless and thread-safe.
    """
    
    def __init__(self, config: TemporalConfig):
        """
        Initialize temporal computer with configuration.
        
        Args:
            config: Temporal configuration containing window sizes and weights
        """
        self.config = config
        logger.info(
            f"Initialized TemporalUrgencyComputer with "
            f"recent_window={config.recent_window_hours}h, "
            f"historical_window={config.historical_window_hours}h"
        )
    
    def compute(self, complaints: List[Complaint], reference_time: datetime = None) -> dict:
        """
        Compute temporal urgency for a cluster.
        
        Args:
            complaints: List of complaints in the cluster
            reference_time: Reference time for "now" (uses current time if None)
            
        Returns:
            Dictionary containing:
                - temporal_urgency: Final temporal score
                - volume_ratio: Recent vs historical volume ratio
                - arrival_rate: Complaints per hour
                - recent_count: Number of recent complaints
                - historical_count: Number of historical complaints
                - mean_inter_arrival: Average time between complaints (seconds)
                
        Raises:
            ValueError: If complaints list is empty
        """
        if not complaints:
            raise ValueError("Cannot compute temporal urgency for empty cluster")
        
        if reference_time is None:
            reference_time = datetime.now()
        
        # Sort complaints by timestamp (handle out-of-order)
        sorted_complaints = sorted(complaints, key=lambda c: c.timestamp)
        
        # Compute volume ratio (burst detection)
        volume_ratio, recent_count, historical_count = self._compute_volume_ratio(
            sorted_complaints, reference_time
        )
        
        # Compute arrival rate (frequency)
        arrival_rate, mean_inter_arrival = self._compute_arrival_rate(sorted_complaints)
        
        # Compute temporal intensity via sigmoid
        temporal_urgency = self._compute_temporal_intensity(volume_ratio, arrival_rate)
        
        logger.debug(
            f"Computed temporal urgency for {len(complaints)} complaints: "
            f"R={volume_ratio:.3f}, λ={arrival_rate:.3f} → T={temporal_urgency:.3f}"
        )
        
        return {
            "temporal_urgency": temporal_urgency,
            "volume_ratio": volume_ratio,
            "arrival_rate": arrival_rate,
            "recent_count": recent_count,
            "historical_count": historical_count,
            "mean_inter_arrival": mean_inter_arrival
        }
    
    def _compute_volume_ratio(
        self, 
        sorted_complaints: List[Complaint], 
        reference_time: datetime
    ) -> Tuple[float, int, int]:
        """
        Compute ratio of recent to historical complaint volume.
        
        Mathematical form:
            R_k = V_recent / (V_historical + ε)
        
        Args:
            sorted_complaints: Complaints sorted by timestamp
            reference_time: Reference time for defining "recent"
            
        Returns:
            Tuple of (volume_ratio, recent_count, historical_count)
        """
        recent_cutoff = reference_time - timedelta(hours=self.config.recent_window_hours)
        historical_cutoff = reference_time - timedelta(hours=self.config.historical_window_hours)
        
        # Count complaints in each window
        recent_count = sum(1 for c in sorted_complaints if c.timestamp >= recent_cutoff)
        historical_count = sum(
            1 for c in sorted_complaints 
            if historical_cutoff <= c.timestamp < recent_cutoff
        )
        
        # Compute ratio with smoothing to prevent division by zero
        volume_ratio = recent_count / (historical_count + self.config.epsilon)
        
        logger.debug(
            f"Volume ratio: {recent_count} recent / {historical_count} historical = {volume_ratio:.3f}"
        )
        
        return volume_ratio, recent_count, historical_count
    
    def _compute_arrival_rate(
        self, 
        sorted_complaints: List[Complaint]
    ) -> Tuple[float, float]:
        """
        Compute complaint arrival rate (complaints per hour).
        
        Mathematical form:
            λ_k = (n-1) / (t_n - t_1)  for n > 1
            λ_k = 0                     for n = 1
        
        Args:
            sorted_complaints: Complaints sorted by timestamp
            
        Returns:
            Tuple of (arrival_rate, mean_inter_arrival_seconds)
            
        Special cases:
            - Single complaint: arrival_rate = 0, mean_inter_arrival = 0
            - Identical timestamps: Uses epsilon to prevent division by zero
        """
        n = len(sorted_complaints)
        
        if n == 1:
            # Single complaint: No inter-arrival time
            logger.debug("Single complaint: arrival rate = 0")
            return 0.0, 0.0
        
        # Compute time span
        first_timestamp = sorted_complaints[0].timestamp
        last_timestamp = sorted_complaints[-1].timestamp
        
        time_span_seconds = (last_timestamp - first_timestamp).total_seconds()
        
        # Handle identical timestamps (all complaints at same time)
        if time_span_seconds < self.config.epsilon:
            logger.debug("Complaints within epsilon time: using high arrival rate")
            # Very high arrival rate for burst
            arrival_rate = 1000.0  # 1000 complaints per hour (effectively instant)
            mean_inter_arrival = self.config.epsilon
        else:
            # Normal case: compute arrival rate
            mean_inter_arrival = time_span_seconds / (n - 1)
            arrival_rate = 3600.0 / mean_inter_arrival  # Convert to per-hour rate
        
        logger.debug(
            f"Arrival rate: {n-1} gaps / {time_span_seconds:.1f}s = {arrival_rate:.3f} per hour"
        )
        
        return arrival_rate, mean_inter_arrival
    
    def _compute_temporal_intensity(self, volume_ratio: float, arrival_rate: float) -> float:
        """
        Compute temporal intensity using sigmoid transformation.
        
        Mathematical form:
            T_k = σ(θ₁·R_k + θ₂·λ_k)
            
            where σ(x) = 1 / (1 + e^(-x))
        
        Args:
            volume_ratio: R_k (recent vs historical volume)
            arrival_rate: λ_k (complaints per hour)
            
        Returns:
            Temporal urgency score in (0, 1)
            
        Numerical stability:
            - Clips input to prevent overflow in exp()
            - Uses numerically stable sigmoid implementation
        """
        # Linear combination of features
        z = (
            self.config.theta_1 * volume_ratio +
            self.config.theta_2 * arrival_rate
        )
        
        # Clip to prevent overflow in exp()
        z = np.clip(z, -self.config.sigmoid_clip_value, self.config.sigmoid_clip_value)
        
        # Numerically stable sigmoid
        temporal_intensity = self._stable_sigmoid(z)
        
        logger.debug(f"Sigmoid input: z={z:.3f} → σ(z)={temporal_intensity:.3f}")
        
        return float(temporal_intensity)
    
    def _stable_sigmoid(self, x: float) -> float:
        """
        Numerically stable sigmoid function.
        
        Standard sigmoid can overflow for large |x|.
        This implementation handles both positive and negative x safely.
        
        Args:
            x: Input value
            
        Returns:
            σ(x) = 1 / (1 + e^(-x))
        """
        if x >= 0:
            # For positive x: σ(x) = 1 / (1 + e^(-x))
            return 1.0 / (1.0 + np.exp(-x))
        else:
            # For negative x: σ(x) = e^x / (1 + e^x)
            # This avoids overflow for large negative x
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)
    
    def compute_multiple(self, cluster_complaints: dict, reference_time: datetime = None) -> dict:
        """
        Compute temporal urgency for multiple clusters.
        
        Args:
            cluster_complaints: Dict mapping cluster_id -> List[Complaint]
            reference_time: Reference time for all computations
            
        Returns:
            Dict mapping cluster_id -> temporal result
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        results = {}
        
        for cluster_id, complaints in cluster_complaints.items():
            try:
                results[cluster_id] = self.compute(complaints, reference_time)
                logger.debug(f"Computed temporal urgency for cluster {cluster_id}")
            except ValueError as e:
                logger.error(f"Failed to compute temporal urgency for cluster {cluster_id}: {e}")
                raise ValueError(f"Cluster {cluster_id}: {e}") from e
        
        logger.info(f"Computed temporal urgency for {len(results)} clusters")
        return results


# Utility function
def compute_temporal_urgency(
    complaints: List[Complaint],
    config: TemporalConfig = None,
    reference_time: datetime = None
) -> float:
    """
    Convenience function to compute temporal urgency.
    
    Args:
        complaints: List of complaints
        config: Optional custom configuration
        reference_time: Optional reference time
        
    Returns:
        Temporal urgency score
    """
    if config is None:
        from decision_engine.config import config as default_config
        config = default_config.temporal
    
    computer = TemporalUrgencyComputer(config)
    result = computer.compute(complaints, reference_time)
    return result["temporal_urgency"]
