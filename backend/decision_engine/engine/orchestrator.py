"""
Decision Engine Orchestrator

This module orchestrates the complete urgency computation pipeline,
coordinating all components:
- Structural aggregation
- Temporal analysis
- Raw urgency fusion
- Calibration
- Summarization

This is the main interface for computing cluster urgencies.

Pipeline Architecture:
---------------------

Input: List[Complaint] with cluster assignments
  ↓
Group by cluster_id
  ↓
For each cluster:
  ├─→ Structural Aggregation  →  U_struct
  ├─→ Temporal Analysis       →  T_k
  └─→ Fusion                  →  U_raw = λ₁·U_struct + λ₂·T_k
       ↓
     Calibration:
       ├─→ Size Normalization →  U_size
       └─→ Smoothing          →  U_final
       ↓
     Summarization            →  ClusterSummary
  ↓
Output: Ranked list of ClusterSummary objects

Design Principles:
-----------------
1. Separation of Concerns: Each component is independent
2. Configurability: All weights/parameters in config
3. Determinism: Same input → same output (for testing)
4. Observability: Full breakdown of computation
5. Extensibility: Easy to add new components
"""

from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import logging

from decision_engine.config import DecisionEngineConfig
from decision_engine.services.data_interface import Complaint
from decision_engine.engine.aggregation import StructuralAggregator
from decision_engine.engine.temporal import TemporalUrgencyComputer
from decision_engine.engine.calibration import UrgencyCalibrator
from decision_engine.engine.summarizer import ClusterSummarizer
from decision_engine.schemas.response_models import (
    ClusterSummary as ClusterSummaryResponse,
    ClusterUrgencyBreakdown
)

logger = logging.getLogger(__name__)


class DecisionEngineOrchestrator:
    """
    Orchestrates the complete urgency computation pipeline.
    
    This is the main entry point for computing cluster urgencies.
    """
    
    def __init__(self, config: DecisionEngineConfig = None):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Engine configuration (uses default if None)
        """
        if config is None:
            from decision_engine.config import config as default_config
            config = default_config
        
        self.config = config
        
        # Initialize all components
        self.aggregator = StructuralAggregator(config.aggregation)
        self.temporal_computer = TemporalUrgencyComputer(config.temporal)
        self.calibrator = UrgencyCalibrator(config.calibration)
        self.summarizer = ClusterSummarizer(config.summarizer)
        
        logger.info("Initialized DecisionEngineOrchestrator with all components")
    
    def process_complaints(
        self, 
        complaints: List[Complaint],
        reference_time: datetime = None
    ) -> List[ClusterSummaryResponse]:
        """
        Process complaints and compute urgencies for all clusters.
        
        This is the main orchestration method that runs the complete pipeline.
        
        Args:
            complaints: List of complaints with cluster assignments
            reference_time: Reference time for temporal analysis (uses now if None)
            
        Returns:
            List of ClusterSummaryResponse objects, sorted by urgency (descending)
            
        Raises:
            ValueError: If input is invalid
        """
        if not complaints:
            raise ValueError("Cannot process empty complaint list")
        
        if reference_time is None:
            reference_time = datetime.now()
        
        logger.info(f"Processing {len(complaints)} complaints")
        
        # Step 1: Group complaints by cluster
        cluster_groups = self._group_by_cluster(complaints)
        logger.info(f"Grouped into {len(cluster_groups)} clusters")
        
        # Step 2: Process each cluster
        results = []
        for cluster_id, cluster_complaints in cluster_groups.items():
            try:
                result = self._process_cluster(
                    cluster_id, 
                    cluster_complaints, 
                    reference_time
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process cluster {cluster_id}: {e}")
                raise
        
        # Step 3: Sort by final urgency (descending)
        results.sort(key=lambda r: r.final_urgency, reverse=True)
        
        logger.info(
            f"Processed {len(results)} clusters, "
            f"top urgency: {results[0].final_urgency:.3f}"
        )
        
        return results
    
    def _group_by_cluster(self, complaints: List[Complaint]) -> Dict[str, List[Complaint]]:
        """
        Group complaints by cluster_id.
        
        Args:
            complaints: List of complaints
            
        Returns:
            Dict mapping cluster_id -> List[Complaint]
        """
        clusters = defaultdict(list)
        for complaint in complaints:
            clusters[complaint.cluster_id].append(complaint)
        return dict(clusters)
    
    def _process_cluster(
        self,
        cluster_id: str,
        complaints: List[Complaint],
        reference_time: datetime
    ) -> ClusterSummaryResponse:
        """
        Process a single cluster through the complete pipeline.
        
        Args:
            cluster_id: Cluster identifier
            complaints: Complaints in this cluster
            reference_time: Reference time for temporal analysis
            
        Returns:
            ClusterSummaryResponse with full breakdown
        """
        logger.debug(f"Processing cluster {cluster_id} with {len(complaints)} complaints")
        
        # Step 1: Structural aggregation
        structural_result = self.aggregator.aggregate(complaints)
        structural_urgency = structural_result["structural_urgency"]
        
        # Step 2: Temporal analysis
        temporal_result = self.temporal_computer.compute(complaints, reference_time)
        temporal_urgency = temporal_result["temporal_urgency"]
        
        # Step 3: Fusion (combine structural and temporal)
        raw_urgency = self._fuse_urgencies(structural_urgency, temporal_urgency)
        
        # Step 4: Calibration (size normalization + smoothing)
        calibration_result = self.calibrator.calibrate(
            cluster_id=cluster_id,
            raw_urgency=raw_urgency,
            complaint_count=len(complaints),
            current_time=reference_time
        )
        
        # Step 5: Summarization
        summary = self.summarizer.summarize(complaints)
        
        # Step 6: Build comprehensive response
        return self._build_response(
            cluster_id=cluster_id,
            complaints=complaints,
            structural_result=structural_result,
            temporal_result=temporal_result,
            raw_urgency=raw_urgency,
            calibration_result=calibration_result,
            summary=summary,
            reference_time=reference_time
        )
    
    def _fuse_urgencies(self, structural: float, temporal: float) -> float:
        """
        Fuse structural and temporal urgencies.
        
        Mathematical form:
            U_raw = λ₁·U_struct + λ₂·T_k
        
        Args:
            structural: Structural urgency component
            temporal: Temporal urgency component
            
        Returns:
            Raw fused urgency
        """
        raw_urgency = (
            self.config.fusion.lambda_1 * structural +
            self.config.fusion.lambda_2 * temporal
        )
        
        # Ensure in valid range
        raw_urgency = max(0.0, min(1.0, raw_urgency))
        
        logger.debug(
            f"Fused urgencies: {self.config.fusion.lambda_1}×{structural:.3f} + "
            f"{self.config.fusion.lambda_2}×{temporal:.3f} = {raw_urgency:.3f}"
        )
        
        return raw_urgency
    
    def _build_response(
        self,
        cluster_id: str,
        complaints: List[Complaint],
        structural_result: dict,
        temporal_result: dict,
        raw_urgency: float,
        calibration_result: dict,
        summary,
        reference_time: datetime
    ) -> ClusterSummaryResponse:
        """
        Build comprehensive response object with full breakdown.
        
        Args:
            cluster_id: Cluster identifier
            complaints: Complaints in cluster
            structural_result: Result from aggregation
            temporal_result: Result from temporal analysis
            raw_urgency: Fused urgency
            calibration_result: Result from calibration
            summary: Cluster summary
            reference_time: Reference time
            
        Returns:
            ClusterSummaryResponse object
        """
        # Get timestamps
        timestamps = [c.timestamp for c in complaints]
        earliest = min(timestamps)
        latest = max(timestamps)
        
        # Build urgency breakdown
        breakdown = ClusterUrgencyBreakdown(
            structural_urgency=structural_result["structural_urgency"],
            temporal_urgency=temporal_result["temporal_urgency"],
            raw_urgency=raw_urgency,
            size_normalized_urgency=calibration_result["size_normalized_urgency"],
            final_urgency=calibration_result["final_urgency"],
            mean_urgency=structural_result["mean_urgency"],
            max_urgency=structural_result["max_urgency"],
            percentile_90_urgency=structural_result["percentile_urgency"],
            volume_ratio=temporal_result["volume_ratio"],
            arrival_rate=temporal_result["arrival_rate"],
            complaint_count=len(complaints),
            previous_urgency=calibration_result["previous_urgency"]
        )
        
        # Build cluster summary response
        return ClusterSummaryResponse(
            cluster_id=cluster_id,
            final_urgency=calibration_result["final_urgency"],
            complaint_count=len(complaints),
            top_complaints=summary.top_complaint_ids,
            summary_text=summary.summary_text,
            breakdown=breakdown,
            earliest_complaint=earliest,
            latest_complaint=latest
        )
    
    def clear_calibration_memory(self, cluster_id: str = None) -> None:
        """
        Clear calibration memory (useful for testing or reset).
        
        Args:
            cluster_id: Specific cluster to clear (None = clear all)
        """
        self.calibrator.clear_memory(cluster_id)
        logger.info(f"Cleared calibration memory for {cluster_id or 'all clusters'}")


# Convenience function
def process_complaints(
    complaints: List[Complaint],
    config: DecisionEngineConfig = None,
    reference_time: datetime = None
) -> List[ClusterSummaryResponse]:
    """
    Convenience function to process complaints.
    
    Args:
        complaints: List of complaints with cluster assignments
        config: Optional custom configuration
        reference_time: Optional reference time
        
    Returns:
        List of ClusterSummaryResponse objects, ranked by urgency
    """
    orchestrator = DecisionEngineOrchestrator(config)
    return orchestrator.process_complaints(complaints, reference_time)
