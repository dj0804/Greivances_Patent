"""
Cluster Summarization Engine

This module generates summaries for complaint clusters by selecting
and aggregating the most urgent complaints.

Design Philosophy:
-----------------
The summarizer is intentionally model-agnostic. It doesn't assume:
- Specific NLP/summarization model
- Transformer architecture
- External API (GPT, etc.)

Instead, it provides:
- Extraction-based summarization (selecting top-k complaints)
- Structured summary objects for downstream processing
- Hooks for integration with summarization models

This allows:
1. Simple concatenation for immediate use
2. Future integration with abstractive summarization
3. Flexibility in model choice

Mathematical Foundation:
-----------------------
Given a cluster k with complaints C = {c₁, c₂, ..., cₙ}:

1. Rank complaints by urgency:
    sorted_C = sort(C, key=urgency_score, descending=True)

2. Select top-k:
    top_k = sorted_C[:k]

3. Generate summary:
    summary = concat(text(c) for c in top_k)

This is extraction-based summarization - selecting salient content
rather than generating new text.

Future Extensions:
-----------------
The summary object can be passed to:
- Abstractive summarization models (T5, BART, GPT)
- Keyword extraction (TF-IDF, TextRank)
- Topic modeling (LDA, BERTopic)
- Sentiment analysis

The modular design supports these extensions without changing core logic.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from decision_engine.config import SummarizerConfig
from decision_engine.services.data_interface import Complaint

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """
    Structured summary of a complaint cluster.
    
    This object is ready for:
    - Direct display
    - API serialization
    - Downstream model input
    """
    cluster_id: str
    top_complaint_ids: List[str]
    summary_text: str
    complaint_count: int
    mean_urgency: float
    max_urgency: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "top_complaint_ids": self.top_complaint_ids,
            "summary_text": self.summary_text,
            "complaint_count": self.complaint_count,
            "mean_urgency": self.mean_urgency,
            "max_urgency": self.max_urgency
        }


class ClusterSummarizer:
    """
    Generates summaries for complaint clusters.
    
    Uses extraction-based approach: selects top-k most urgent complaints
    and concatenates their text.
    """
    
    def __init__(self, config: SummarizerConfig):
        """
        Initialize summarizer with configuration.
        
        Args:
            config: Summarizer configuration
        """
        self.config = config
        logger.info(
            f"Initialized ClusterSummarizer with top_k={config.top_k_complaints}"
        )
    
    def summarize(self, complaints: List[Complaint]) -> ClusterSummary:
        """
        Generate summary for a cluster of complaints.
        
        Args:
            complaints: List of complaints in the cluster
            
        Returns:
            ClusterSummary object containing top complaints and concatenated text
            
        Raises:
            ValueError: If complaints list is empty or cluster_ids don't match
        """
        if not complaints:
            raise ValueError("Cannot summarize empty cluster")
        
        # Validate all complaints are from same cluster
        cluster_ids = set(c.cluster_id for c in complaints)
        if len(cluster_ids) > 1:
            raise ValueError(
                f"All complaints must be from same cluster, got {cluster_ids}"
            )
        cluster_id = complaints[0].cluster_id
        
        # Select top-k most urgent complaints
        top_complaints = self._select_top_k(complaints)
        
        # Generate concatenated summary text
        summary_text = self._generate_summary_text(top_complaints)
        
        # Compute statistics
        urgency_scores = [c.urgency_score for c in complaints]
        mean_urgency = sum(urgency_scores) / len(urgency_scores)
        max_urgency = max(urgency_scores)
        
        logger.debug(
            f"Summarized cluster {cluster_id}: "
            f"{len(top_complaints)}/{len(complaints)} complaints, "
            f"{len(summary_text)} chars"
        )
        
        return ClusterSummary(
            cluster_id=cluster_id,
            top_complaint_ids=[c.complaint_id for c in top_complaints],
            summary_text=summary_text,
            complaint_count=len(complaints),
            mean_urgency=mean_urgency,
            max_urgency=max_urgency
        )
    
    def _select_top_k(self, complaints: List[Complaint]) -> List[Complaint]:
        """
        Select top-k complaints by urgency score.
        
        Args:
            complaints: List of complaints
            
        Returns:
            Top-k complaints (or all if fewer than k)
        """
        # Sort by urgency (descending)
        sorted_complaints = sorted(
            complaints, 
            key=lambda c: c.urgency_score, 
            reverse=True
        )
        
        # Select top k
        k = min(self.config.top_k_complaints, len(sorted_complaints))
        top_k = sorted_complaints[:k]
        
        logger.debug(f"Selected top {k} complaints from {len(complaints)}")
        
        return top_k
    
    def _generate_summary_text(self, complaints: List[Complaint]) -> str:
        """
        Generate summary text from top complaints.
        
        Currently uses simple concatenation. Can be extended with:
        - Abstractive summarization models
        - Template-based generation
        - Hierarchical summarization
        
        Args:
            complaints: Top-k complaints
            
        Returns:
            Summary text (truncated to max length)
        """
        # Concatenate complaint texts with separators
        texts = [
            f"[{c.complaint_id}] {c.text.strip()}"
            for c in complaints
        ]
        
        full_text = " | ".join(texts)
        
        # Truncate if exceeds max length
        if len(full_text) > self.config.max_summary_length:
            truncated = full_text[:self.config.max_summary_length - 3] + "..."
            logger.debug(
                f"Truncated summary from {len(full_text)} to {len(truncated)} chars"
            )
            return truncated
        
        return full_text
    
    def summarize_multiple(self, cluster_complaints: Dict[str, List[Complaint]]) -> Dict[str, ClusterSummary]:
        """
        Generate summaries for multiple clusters.
        
        Args:
            cluster_complaints: Dict mapping cluster_id -> List[Complaint]
            
        Returns:
            Dict mapping cluster_id -> ClusterSummary
        """
        summaries = {}
        
        for cluster_id, complaints in cluster_complaints.items():
            try:
                summaries[cluster_id] = self.summarize(complaints)
                logger.debug(f"Summarized cluster {cluster_id}")
            except ValueError as e:
                logger.error(f"Failed to summarize cluster {cluster_id}: {e}")
                raise
        
        logger.info(f"Summarized {len(summaries)} clusters")
        return summaries


# Utility function
def summarize_cluster(
    complaints: List[Complaint],
    config: SummarizerConfig = None
) -> ClusterSummary:
    """
    Convenience function to summarize a cluster.
    
    Args:
        complaints: List of complaints in the cluster
        config: Optional custom configuration
        
    Returns:
        ClusterSummary object
    """
    if config is None:
        from decision_engine.config import config as default_config
        config = default_config.summarizer
    
    summarizer = ClusterSummarizer(config)
    return summarizer.summarize(complaints)


# Advanced summarization hooks (for future extension)
class AbstractiveSummarizer:
    """
    PLACEHOLDER: Hook for abstractive summarization using transformers.
    
    This would integrate with models like:
    - T5 (Google)
    - BART (Facebook)
    - Pegasus (Google)
    - GPT-based summarization
    
    Not implemented to keep Decision Engine model-agnostic.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """Initialize with summarization model."""
        raise NotImplementedError("Abstractive summarization not yet implemented")
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """
        Generate abstractive summary.
        
        Would use:
        - Tokenization
        - Model inference
        - Beam search / sampling
        - Post-processing
        """
        raise NotImplementedError("Abstractive summarization not yet implemented")


class KeywordExtractor:
    """
    PLACEHOLDER: Hook for keyword extraction.
    
    Could use:
    - TF-IDF
    - TextRank
    - RAKE
    - YAKE
    - KeyBERT
    
    Not implemented to keep engine lightweight.
    """
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top-k keywords from text."""
        raise NotImplementedError("Keyword extraction not yet implemented")
