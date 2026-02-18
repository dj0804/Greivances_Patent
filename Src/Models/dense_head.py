"""
Dense head for urgency prediction with temporal features.
"""

from tensorflow import keras
from tensorflow.keras import layers


class DenseHead:
    """
    Dense neural network head for combining text features with temporal data.
    """
    
    def __init__(
        self,
        input_dim: int,
        temporal_dim: int,
        hidden_units: list = [128, 64],
        num_classes: int = 3,
        dropout: float = 0.5
    ):
        """
        Initialize dense head.
        
        Args:
            input_dim: Dimension of text features
            temporal_dim: Dimension of temporal features
            hidden_units: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the dense head architecture."""
        # Text features input
        text_input = layers.Input(shape=(self.input_dim,), name='text_features')
        
        # Temporal features input
        temporal_input = layers.Input(
            shape=(self.temporal_dim,),
            name='temporal_features'
        )
        
        # Concatenate features
        x = layers.Concatenate()([text_input, temporal_input])
        
        # Dense layers
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(
            inputs=[text_input, temporal_input],
            outputs=outputs
        )
        
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
