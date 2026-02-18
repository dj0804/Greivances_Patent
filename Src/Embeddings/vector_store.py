"""
Vector database interface for storing and retrieving embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path


class VectorStore:
    """
    Simple vector store for complaint embeddings.
    Can be extended to use FAISS, Pinecone, or other vector databases.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize vector store.
        
        Args:
            dimension: Dimensionality of vectors
        """
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
        self.ids = []
    
    def add(
        self,
        vector: np.ndarray,
        id: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a vector to the store.
        
        Args:
            vector: Embedding vector
            id: Unique identifier
            metadata: Associated metadata
        """
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} "
                f"doesn't match store dimension {self.dimension}"
            )
        
        self.vectors.append(vector)
        self.ids.append(id)
        self.metadata.append(metadata or {})
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find most similar vectors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        if not self.vectors:
            return []
        
        vectors_array = np.array(self.vectors)
        
        # Cosine similarity
        query_norm = query_vector / np.linalg.norm(query_vector)
        vectors_norm = vectors_array / np.linalg.norm(
            vectors_array, axis=1, keepdims=True
        )
        
        similarities = np.dot(vectors_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            (self.ids[i], similarities[i], self.metadata[i])
            for i in top_indices
        ]
        
        return results
    
    def save(self, path: Path) -> None:
        """Save vector store to disk."""
        data = {
            "dimension": self.dimension,
            "vectors": self.vectors,
            "ids": self.ids,
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load vector store from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        store = cls(data["dimension"])
        store.vectors = data["vectors"]
        store.ids = data["ids"]
        store.metadata = data["metadata"]
        
        return store
