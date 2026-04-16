"""
CNN model for text classification.
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import numpy as np


class CNNModel:
    """
    Convolutional Neural Network for text classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        max_length: int = 100,
        filters: int = 128,
        kernel_size: int = 5,
        num_classes: int = 3,
        dropout: float = 0.5,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize CNN model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            max_length: Maximum sequence length
            filters: Number of CNN filters
            kernel_size: Size of convolution kernel
            num_classes: Number of output classes
            dropout: Dropout rate
            embedding_matrix: Pretrained embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.embedding_matrix = embedding_matrix
        
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the CNN architecture."""
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        if self.embedding_matrix is not None:
            x = layers.Embedding(
                self.vocab_size,
                self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_length,
                trainable=False
            )(inputs)
        else:
            x = layers.Embedding(
                self.vocab_size,
                self.embedding_dim,
                input_length=self.max_length
            )(inputs)
        
        # CNN layers
        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            activation='relu'
        )(x)
        x = layers.GlobalMaxPooling1D()(x)
        
        # Dense layers
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile(
        self,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: list = None
    ):
        """Compile the model."""
        if metrics is None:
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_model(self) -> keras.Model:
        """Get the Keras model."""
        return self.model
