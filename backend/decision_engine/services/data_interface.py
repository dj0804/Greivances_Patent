"""
Data Interface Layer - Clustering Source Abstraction

This module provides an abstract interface for accessing complaint data,
ensuring the Decision Engine is completely agnostic to the clustering source
(FAISS, VectorDB, or any other mechanism).

This is the CRITICAL abstraction layer that enables clustering-source independence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Protocol
import logging

logger = logging.getLogger(__name__)


@dataclass
class Complaint:
    """
    Core complaint data structure.
    
    This is the ONLY data structure the Decision Engine operates on.
    All clustering sources must provide data in this format.
    
    Attributes:
        complaint_id: Unique identifier for the complaint
        cluster_id: Cluster assignment (source-agnostic)
        urgency_score: Complaint-level urgency [0, 1]
        timeline_score: Timeline-based urgency factor [0, 1]
        timestamp: When the complaint was submitted
        text: Complaint text content
    """
    complaint_id: str
    cluster_id: str
    urgency_score: float
    timeline_score: float
    timestamp: datetime
    text: str
    
    def __post_init__(self):
        """Validate complaint data."""
        if not (0.0 <= self.urgency_score <= 1.0):
            raise ValueError(
                f"urgency_score must be in [0, 1], got {self.urgency_score}"
            )
        if not (0.0 <= self.timeline_score <= 1.0):
            raise ValueError(
                f"timeline_score must be in [0, 1], got {self.timeline_score}"
            )
        if not self.complaint_id:
            raise ValueError("complaint_id cannot be empty")
        if not self.cluster_id:
            raise ValueError("cluster_id cannot be empty")
        if not self.text:
            raise ValueError("text cannot be empty")


class ComplaintDataSource(Protocol):
    """
    Protocol defining the interface for complaint data sources.
    
    Any clustering mechanism (FAISS, VectorDB, CSV, etc.) must implement
    this protocol to be compatible with the Decision Engine.
    
    This is a Protocol (structural typing), not an ABC, allowing maximum
    flexibility in implementation while ensuring type safety.
    """
    
    def fetch_all_complaints(self) -> List[Complaint]:
        """
        Retrieve all complaints from the data source.
        
        Returns:
            List of all complaints with cluster assignments
            
        Raises:
            DataSourceError: If data cannot be retrieved
        """
        ...
    
    def fetch_by_cluster(self, cluster_id: str) -> List[Complaint]:
        """
        Retrieve all complaints belonging to a specific cluster.
        
        Args:
            cluster_id: The cluster identifier
            
        Returns:
            List of complaints in the specified cluster
            
        Raises:
            DataSourceError: If data cannot be retrieved
        """
        ...


class InMemoryComplaintDataSource:
    """
    In-memory implementation of ComplaintDataSource for testing and development.
    
    This serves as:
    1. Reference implementation
    2. Testing mock
    3. Standalone mode for the Decision Engine
    """
    
    def __init__(self, complaints: List[Complaint] = None):
        """
        Initialize with optional list of complaints.
        
        Args:
            complaints: Pre-loaded complaints. Empty list if None.
        """
        self._complaints: Dict[str, Complaint] = {}
        self._cluster_index: Dict[str, List[str]] = {}
        
        if complaints:
            for complaint in complaints:
                self.add_complaint(complaint)
        
        logger.info(f"Initialized InMemoryComplaintDataSource with {len(self._complaints)} complaints")
    
    def add_complaint(self, complaint: Complaint) -> None:
        """
        Add a complaint to the in-memory store.
        
        Args:
            complaint: Complaint to add
        """
        self._complaints[complaint.complaint_id] = complaint
        
        # Update cluster index
        if complaint.cluster_id not in self._cluster_index:
            self._cluster_index[complaint.cluster_id] = []
        self._cluster_index[complaint.cluster_id].append(complaint.complaint_id)
        
        logger.debug(f"Added complaint {complaint.complaint_id} to cluster {complaint.cluster_id}")
    
    def fetch_all_complaints(self) -> List[Complaint]:
        """Retrieve all complaints."""
        complaints = list(self._complaints.values())
        logger.info(f"Fetched {len(complaints)} complaints")
        return complaints
    
    def fetch_by_cluster(self, cluster_id: str) -> List[Complaint]:
        """Retrieve complaints for a specific cluster."""
        complaint_ids = self._cluster_index.get(cluster_id, [])
        complaints = [self._complaints[cid] for cid in complaint_ids]
        logger.info(f"Fetched {len(complaints)} complaints for cluster {cluster_id}")
        return complaints
    
    def get_cluster_ids(self) -> List[str]:
        """Get all unique cluster IDs."""
        return list(self._cluster_index.keys())
    
    def clear(self) -> None:
        """Clear all data."""
        self._complaints.clear()
        self._cluster_index.clear()
        logger.info("Cleared all complaints from data source")


class DataSourceError(Exception):
    """Exception raised when data source operations fail."""
    pass


# Example adapter pattern for future FAISS integration
class FAISSComplaintDataSourceAdapter:
    """
    PLACEHOLDER: Adapter for FAISS-based clustering.
    
    This shows HOW to integrate with FAISS without implementing it directly.
    The Decision Engine never imports FAISS directly.
    """
    
    def __init__(self, faiss_index, complaint_store):
        """
        Args:
            faiss_index: FAISS index object (opaque to Decision Engine)
            complaint_store: Store mapping complaint_id -> raw complaint data
        """
        self.faiss_index = faiss_index
        self.complaint_store = complaint_store
    
    def fetch_all_complaints(self) -> List[Complaint]:
        """
        Fetch all complaints with their FAISS cluster assignments.
        
        Implementation would:
        1. Query FAISS for all cluster assignments
        2. Transform to Complaint objects
        3. Return list
        """
        raise NotImplementedError("FAISS adapter not yet implemented")
    
    def fetch_by_cluster(self, cluster_id: str) -> List[Complaint]:
        """
        Fetch complaints from a specific FAISS cluster.
        
        Implementation would:
        1. Query FAISS for cluster members
        2. Retrieve complaint data
        3. Transform to Complaint objects
        4. Return list
        """
        raise NotImplementedError("FAISS adapter not yet implemented")


# Example adapter for VectorDB
class VectorDBComplaintDataSourceAdapter:
    """
    PLACEHOLDER: Adapter for VectorDB-based clustering.
    
    Similar pattern to FAISS adapter.
    """
    
    def __init__(self, vector_db_client, collection_name: str):
        """
        Args:
            vector_db_client: VectorDB client (e.g., Pinecone, Weaviate)
            collection_name: Name of the complaint collection
        """
        self.client = vector_db_client
        self.collection_name = collection_name
    
    def fetch_all_complaints(self) -> List[Complaint]:
        """Fetch all complaints with VectorDB cluster assignments."""
        raise NotImplementedError("VectorDB adapter not yet implemented")
    
    def fetch_by_cluster(self, cluster_id: str) -> List[Complaint]:
        """Fetch complaints from a specific VectorDB cluster."""
        raise NotImplementedError("VectorDB adapter not yet implemented")
