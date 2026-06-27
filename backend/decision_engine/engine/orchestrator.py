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
import math

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
    
    def __init__(self, config: DecisionEngineConfig = None, db_manager=None):
        """
        Initialize orchestrator with configuration.

        Args:
            config: Engine configuration (uses default if None)
            db_manager: Optional HostelDB instance for H(g) historical calibration
        """
        if config is None:
            from decision_engine.config import config as default_config
            config = default_config

        self.config = config
        self.db_manager = db_manager

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
        
        # Step 3: Compute additional fusion signals
        d_g = self._compute_cluster_density(complaints)
        s_g = self._compute_semantic_similarity(complaints)
        h_g = self._compute_historical_calibration(cluster_id)

        # Step 3b: Fusion (combine all signals)
        raw_urgency = self._fuse_urgencies(structural_urgency, temporal_urgency, d_g, s_g, h_g)

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
            reference_time=reference_time,
            d_g=d_g,
            s_g=s_g,
            h_g=h_g,
        )
    
    def _compute_cluster_density(self, complaints: List[Complaint]) -> float:
        """
        Compute D(g): cluster density signal.

        D_g = spatial_weight * spatial_density + temporal_weight * temporal_density

        spatial_density = 1 / (1 + mean_pairwise_urgency_distance)
        temporal_density = min(1.0, n / 10)
        """
        cfg = self.config.cluster_density
        n = len(complaints)

        if n <= 1:
            spatial_density = 1.0
        else:
            scores = [c.urgency_score for c in complaints]
            diffs = []
            for i in range(n):
                for j in range(i + 1, n):
                    diffs.append(abs(scores[i] - scores[j]))
            mean_dist = sum(diffs) / len(diffs)
            spatial_density = 1.0 / (1.0 + mean_dist)

        temporal_density = min(1.0, n / 10.0)

        d_g = cfg.spatial_weight * spatial_density + cfg.temporal_weight * temporal_density
        logger.debug(f"D(g) = {d_g:.3f} (spatial={spatial_density:.3f}, temporal={temporal_density:.3f})")
        return float(d_g)

    def _compute_semantic_similarity(self, complaints: List[Complaint]) -> float:
        """
        Compute S(g): mean urgency_score of top-3 highest-urgency complaints.

        Used as a proxy for similarity to known high-urgency cases.
        """
        if len(complaints) == 1:
            return float(complaints[0].urgency_score)
        top_k = sorted(complaints, key=lambda c: c.urgency_score, reverse=True)[:3]
        s_g = sum(c.urgency_score for c in top_k) / len(top_k)
        logger.debug(f"S(g) = {s_g:.3f}")
        return float(s_g)

    def _compute_historical_calibration(self, cluster_id: str) -> float:
        """
        Compute H(g): historical SLA breach rate for this cluster.

        Returns 0.5 (neutral prior) when no db_manager is configured or on error.
        """
        if self.db_manager is None:
            return 0.5
        try:
            return float(self.db_manager.get_sla_breach_rate(cluster_id))
        except Exception as exc:
            logger.warning("Failed to compute H(g) for cluster %s: %s", cluster_id, exc)
            return 0.5

    def _fuse_urgencies(
        self,
        structural: float,
        temporal: float,
        d_g: float = 0.0,
        s_g: float = 0.0,
        h_g: float = 0.0,
    ) -> float:
        """
        Fuse all urgency signals.

        Mathematical form:
            U_raw = λ₁·U_struct + λ₂·T_k + λ₃·D_g + λ₄·S_g + λ₅·H_g
        """
        raw_urgency = (
            self.config.fusion.lambda_1 * structural
            + self.config.fusion.lambda_2 * temporal
            + self.config.fusion.lambda_3 * d_g
            + self.config.fusion.lambda_4 * s_g
            + self.config.fusion.lambda_5 * h_g
        )

        raw_urgency = max(0.0, min(1.0, raw_urgency))

        logger.debug(
            f"Fused urgencies: λ1×{structural:.3f} + λ2×{temporal:.3f} + "
            f"λ3×{d_g:.3f} + λ4×{s_g:.3f} + λ5×{h_g:.3f} = {raw_urgency:.3f}"
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
        reference_time: datetime,
        d_g: float = 0.0,
        s_g: float = 0.0,
        h_g: float = 0.0,
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
            previous_urgency=calibration_result["previous_urgency"],
            cluster_density=d_g,
            semantic_similarity=s_g,
            historical_calibration=h_g,
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
