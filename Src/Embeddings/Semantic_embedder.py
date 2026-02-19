"""
Semantic embedding using FastText pretrained models.
"""

import numpy as np
from gensim.models import fasttext
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class FastTextEncoder:
    """
    FastText-based semantic encoder for complaint text.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize FastText encoder.
        
        Args:
            model_path: Path to pretrained FastText model
        """
        print("Loading FastText model (this may take a minute)")
        self.model = fasttext.load_facebook_model(model_path)
        self.vector_size = self.model.vector_size

    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Convert tokens to embedding vectors.
        
        Args:
            tokens: List of tokens to embed
            
        Returns:
            Array of embedding vectors
        """
        vectors = [self.model.wv[token] for token in tokens]
        return np.array(vectors)
    
    def encode_sentence(self, tokens: List[str]) -> np.ndarray:
        """
        Get sentence-level embedding by averaging token embeddings.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Single averaged embedding vector
        """
        if not tokens:
            return np.zeros(self.vector_size)
        
        vectors = self.encode_tokens(tokens)
        return np.mean(vectors, axis=0)

    def get_embedding_matrix(
        self,
        word_index: Dict[str, int],
        max_features: int = None
    ) -> np.ndarray:
        """
        Create embedding matrix for Keras Embedding layer.
        
        Args:
            word_index: Dictionary mapping words to indices
            max_features: Maximum vocabulary size
            
        Returns:
            Embedding matrix for neural network
        """
        vocab_size = len(word_index) + 1
        if max_features:
            vocab_size = min(vocab_size, max_features + 1)
        
        embedding_matrix = np.zeros((vocab_size, self.vector_size))
        
        for word, i in word_index.items():
            if i >= vocab_size:
                continue
                
            if word in self.model.wv:
                embedding_matrix[i] = self.model.wv[word]
            else:
                # Use FastText's subword handling for OOV words
                embedding_matrix[i] = self.model.wv.get_vector(word)
                
        return embedding_matrix

class Semantic:
    def __init__(self):
            self.model = SentenceTransformer('../Models/sbert_local_model')

    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences into semantic vectors using SBERT.
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            Array of semantic embedding vectors
        """
        return self.model.encode(sentences)
        
Se=Semantic()
li=(Se.encode(["This is a test sentence.", "Another example sentence."]))
print(li.shape)
