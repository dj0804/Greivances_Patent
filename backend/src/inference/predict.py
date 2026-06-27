"""
End-to-end prediction pipeline for grievance urgency classification.
"""

import pickle
import logging
from tensorflow import keras
try:
    from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
except ImportError:
    KerasTokenizer = None
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from ..preprocessing import TextCleaner, Tokenise
from ..preprocessing.temporal_encoder import TemporalEncoder
from ..embeddings import FastTextEncoder

_logger = logging.getLogger(__name__)


class GrievancePredictor:
    """
    Complete pipeline for predicting grievance urgency.
    """

    _warned_no_keras_tokenizer = False

    def __init__(
        self,
        model_path: Path,
        fasttext_model_path: Path = None,
        max_length: int = 100,
        tokenizer_path: Path = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model
            fasttext_model_path: Path to FastText model (optional)
            max_length: Maximum sequence length
            tokenizer_path: Path to pickled Keras tokenizer (optional)
        """
        self.model = keras.models.load_model(model_path)
        self.max_length = max_length

        # Initialize preprocessing components
        self.text_cleaner = TextCleaner()
        self.tokenizer = Tokenise()
        self.temporal_encoder = TemporalEncoder()

        # Load fitted Keras tokenizer if provided
        self.keras_tokenizer = None
        if tokenizer_path is not None and Path(tokenizer_path).exists():
            with open(Path(tokenizer_path), 'rb') as f:
                self.keras_tokenizer = pickle.load(f)

        # Initialize FastText encoder if provided
        self.fasttext_encoder = None
        if fasttext_model_path:
            self.fasttext_encoder = FastTextEncoder(str(fasttext_model_path))

        # Urgency labels
        self.urgency_labels = ['Low', 'Medium', 'High']
    
    def predict(
        self,
        complaint_text: str,
        timestamp: datetime = None,
        hostel_id: str = None,
        complaint_type: str = None,
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        """
        Predict urgency for a single complaint.
        
        Args:
            complaint_text: Raw complaint text
            timestamp: When complaint was filed
            hostel_id: Hostel identifier
            complaint_type: Type of complaint
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess text
        cleaned_text = self.text_cleaner.clean(complaint_text)
        tokens = self.tokenizer.tokenise(cleaned_text)
        
        # Encode temporal features if timestamp provided
        temporal_features = None
        if timestamp:
            temporal_features = self.temporal_encoder.encode(
                timestamp, hostel_id, complaint_type
            )
        
        # Convert tokens to sequences (simplified - in production use tokenizer.texts_to_sequences)
        # This is a placeholder; implement proper sequence handling
        sequence = self._tokens_to_sequence(tokens)
        
        # Pad sequence
        padded_sequence = self._pad_sequence(sequence)
        
        # Make prediction
        probabilities = self.model.predict(np.array([padded_sequence]), verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        urgency_level = self.urgency_labels[predicted_class]
        confidence = float(probabilities[predicted_class])
        
        result = {
            'urgency_level': urgency_level,
            'confidence': confidence,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
        }
        
        if temporal_features:
            result['temporal_features'] = temporal_features
        
        if return_probabilities:
            result['probabilities'] = {
                label: float(prob)
                for label, prob in zip(self.urgency_labels, probabilities)
            }
        
        return result
    
    def predict_batch(
        self,
        complaints: List[str],
        timestamps: List[datetime] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Predict urgency for multiple complaints.
        
        Args:
            complaints: List of complaint texts
            timestamps: List of timestamps
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of prediction results
        """
        if timestamps is None:
            timestamps = [None] * len(complaints)
        
        results = []
        for complaint, timestamp in zip(complaints, timestamps):
            result = self.predict(complaint, timestamp, **kwargs)
            results.append(result)
        
        return results
    
    def _tokens_to_sequence(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to integer sequence using the fitted Keras tokenizer.
        Falls back to a hash-based mapping if no tokenizer is loaded.
        """
        if self.keras_tokenizer is not None:
            return self.keras_tokenizer.texts_to_sequences([" ".join(tokens)])[0]
        # Fallback: hash-based mapping (imprecise — load a fitted tokenizer for accurate inference)
        if not GrievancePredictor._warned_no_keras_tokenizer:
            _logger.warning(
                "No fitted Keras tokenizer loaded; using hash fallback. "
                "Pass tokenizer_path= to GrievancePredictor for accurate predictions."
            )
            GrievancePredictor._warned_no_keras_tokenizer = True
        return [abs(hash(token)) % 10000 for token in tokens]
    
    def _pad_sequence(self, sequence: List[int]) -> np.ndarray:
        """Pad or truncate sequence to max_length."""
        if len(sequence) > self.max_length:
            return np.array(sequence[:self.max_length])
        else:
            padded = np.zeros(self.max_length, dtype=int)
            padded[:len(sequence)] = sequence
            return padded
