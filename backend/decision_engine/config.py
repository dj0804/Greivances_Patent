"""
Configuration for Decision Engine

All weights, thresholds, and hyperparameters for urgency computation.
Designed to be easily tunable without code changes.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class AggregationConfig:
    """Configuration for structural urgency aggregation.
    
    Weights for combining mean, max, and percentile of complaint urgencies
    within a cluster. Must sum to 1.0 for interpretability.
    
    Mathematical form:
        U_struct = alpha * mean(U_i) + beta * max(U_i) + delta * percentile_90(U_i)
    """
    alpha: float = 0.4  # Weight for mean urgency
    beta: float = 0.3   # Weight for max urgency
    delta: float = 0.3  # Weight for 90th percentile
    
    percentile_threshold: float = 90.0  # Percentile to compute (0-100)
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.alpha + self.beta + self.delta
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Aggregation weights must sum to 1.0, got {total:.4f}"
            )


@dataclass
class TemporalConfig:
    """Configuration for temporal urgency computation.
    
    Parameters for computing urgency based on complaint arrival patterns,
    volume ratios, and burst detection.
    
    Mathematical form:
        R_k = V_recent / (V_historical + epsilon)
        lambda_k = 1 / mean_inter_arrival_time
        T_k = sigmoid(theta_1 * R_k + theta_2 * lambda_k)
    """
    theta_1: float = 2.0  # Weight for volume ratio in sigmoid
    theta_2: float = 1.5  # Weight for arrival rate in sigmoid
    
    recent_window_hours: float = 24.0  # Hours considered "recent"
    historical_window_hours: float = 168.0  # 1 week historical context
    
    epsilon: float = 1e-6  # Smoothing term to prevent division by zero
    
    # Sigmoid numerical stability
    sigmoid_clip_value: float = 30.0  # Clip inputs to prevent overflow


@dataclass
class CalibrationConfig:
    """Configuration for urgency calibration and smoothing.
    
    Parameters for size normalization and temporal smoothing to prevent
    large clusters from dominating and to reduce volatility.
    
    Mathematical form:
        U_size = U_raw / (1 + log(1 + n_k))
        U_final = gamma * U_prev + (1 - gamma) * U_size
    """
    gamma: float = 0.7  # Exponential smoothing factor (0 = no memory, 1 = no update)
    
    size_penalty_base: float = 1.0  # Base for logarithmic size penalty
    
    # Smoothing behavior
    enable_smoothing: bool = True
    memory_persistence_hours: float = 72.0  # How long to keep previous urgencies
    
    def __post_init__(self):
        """Validate gamma is in valid range."""
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(
                f"Smoothing factor gamma must be in [0, 1], got {self.gamma}"
            )


@dataclass
class FusionConfig:
    """Configuration for fusing structural and temporal urgencies.
    
    Weights for combining structural and temporal components into
    raw urgency score.
    
    Mathematical form:
        U_raw = lambda_1 * U_struct + lambda_2 * T_k
    """
    lambda_1: float = 0.6  # Weight for structural urgency
    lambda_2: float = 0.4  # Weight for temporal urgency
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.lambda_1 + self.lambda_2
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total:.4f}"
            )


@dataclass
class SummarizerConfig:
    """Configuration for cluster summarization."""
    top_k_complaints: int = 5  # Number of top complaints to include in summary
    max_summary_length: int = 500  # Maximum characters for concatenated summary


@dataclass
class DecisionEngineConfig:
    """Master configuration for the entire Decision Engine."""
    
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    
    # Logging
    log_level: str = "INFO"
    
    # Validation
    min_urgency: float = 0.0
    max_urgency: float = 1.0
    
    @classmethod
    def from_env(cls) -> "DecisionEngineConfig":
        """Load configuration from environment variables if available."""
        # This can be extended to read from env vars or config files
        return cls()
    
    def validate_urgency_score(self, score: float) -> bool:
        """Check if urgency score is within valid bounds."""
        return self.min_urgency <= score <= self.max_urgency


# Global configuration instance
config = DecisionEngineConfig()
