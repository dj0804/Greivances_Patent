"""
Combined CNN-BiLSTM model for enhanced feature extraction.
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional
import numpy as np


class CNNBiLSTMModel:
    """
    Hybrid CNN-BiLSTM model combining local and sequential features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        max_length: int = 100,
        cnn_filters: int = 128,
        cnn_kernel_size: int = 5,
        lstm_units: int = 64,
        num_classes: int = 3,
        dropout: float = 0.5,
        embedding_matrix: Optional[np.ndarray] = None,
        use_attention: bool = True,
    ):
        """
        Initialize CNN-BiLSTM model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            max_length: Maximum sequence length
            cnn_filters: Number of CNN filters
            cnn_kernel_size: Size of convolution kernel
            lstm_units: Number of LSTM units
            num_classes: Number of output classes
            dropout: Dropout rate
            embedding_matrix: Pretrained embeddings
            use_attention: Whether to add a self-attention layer after BiLSTM
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.dropout = dropout
        self.embedding_matrix = embedding_matrix
        self.use_attention = use_attention

        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """
        Build the combined CNN-BiLSTM architecture.

        When use_attention=True, a self-attention layer is inserted after the
        second BiLSTM to improve both accuracy and interpretability.
        GlobalAveragePooling1D then reduces the attended sequence to a fixed
        vector before the Dense head. When use_attention=False, GAP is applied
        directly to the BiLSTM sequence output.
        """
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

        # CNN layer for local feature extraction
        x = layers.Conv1D(
            self.cnn_filters,
            self.cnn_kernel_size,
            activation='relu',
            padding='same'
        )(x)
        x = layers.Dropout(self.dropout)(x)

        # First BiLSTM layer for sequential dependencies
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(x)
        x = layers.Dropout(self.dropout)(x)

        # Second BiLSTM — always return_sequences so attention/GAP can consume it
        bilstm_out = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True)
        )(x)
        bilstm_out = layers.Dropout(self.dropout)(bilstm_out)

        if self.use_attention:
            # Self-attention: project bilstm output to query, then attend
            query = layers.Dense(self.lstm_units)(bilstm_out)
            attended = layers.Attention()([query, bilstm_out])
            x = layers.GlobalAveragePooling1D()(attended)
        else:
            x = layers.GlobalAveragePooling1D()(bilstm_out)

        # Dense head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
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


if __name__ == "__main__":
    model = CNNBiLSTMModel(vocab_size=10000)
    model.get_model().summary()
