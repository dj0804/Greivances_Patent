"""
Vector database interface for storing and retrieving embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import faiss
import os


class VectorStore:
    """
    Simple vector store for complaint embeddings.
    Can be extended to use FAISS, Pinecone, or other vector databases.
    """
    
    def __init__(self, dimension: int, autosave_dir: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            dimension: Dimensionality of vectors
            autosave_dir: Directory to automatically save indices
        """
        self.dimension = dimension
        
        # We use Cosine Similarity, so we normalize vectors and use Inner Product
        self.index = faiss.IndexFlatIP(dimension)
        
        self.metadata = []
        self.ids = []
        
        self.autosave_dir = autosave_dir
        if self.autosave_dir:
            os.makedirs(self.autosave_dir, exist_ok=True)
    
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
        
        # FAISS expects 2D array for addition
        vector_2d = np.array([vector], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(vector_2d)
        
        self.index.add(vector_2d)
        
        self.ids.append(id)
        self.metadata.append(metadata or {})
        
        if self.autosave_dir:
            self.save(Path(self.autosave_dir) / "latest")
    
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
        if self.index.ntotal == 0:
            return []
        
        # FAISS expects 2D array
        query_2d = np.array([query_vector], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_2d)
        
        # Search
        similarities, indices = self.index.search(query_2d, top_k)
        
        results = []
        for j, i in enumerate(indices[0]):
            if i != -1: # -1 means not enough results were found
                results.append((self.ids[i], float(similarities[0][j]), self.metadata[i]))
        
        return results
    
    def save(self, path: Path) -> None:
        """Save vector store to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata and IDs
        data = {
            "dimension": self.dimension,
            "ids": self.ids,
            "metadata": self.metadata,
        }
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: Path, autosave_dir: Optional[str] = None) -> "VectorStore":
        """Load vector store from disk."""
        store = cls(0, autosave_dir=autosave_dir) # temporary dimension, will be overwritten
        
        # Load FAISS index
        store.index = faiss.read_index(f"{path}.faiss")
        store.dimension = store.index.d
        
        # Load metadata and IDs
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
        
        store.ids = data["ids"]
        store.metadata = data["metadata"]
        
        return store
