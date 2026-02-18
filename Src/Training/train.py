"""
Model training pipeline.
"""

from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime


class ModelTrainer:
    """
    Handles model training with callbacks and logging.
    """
    
    def __init__(
        self,
        model: keras.Model,
        model_dir: Path,
        log_dir: Path
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            model_dir: Directory to save model checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        **kwargs
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            epochs: Number of epochs
            patience: Early stopping patience
            **kwargs: Additional arguments for model.fit()
            
        Returns:
            Training history
        """
        # Setup callbacks
        callbacks = self._get_callbacks(patience)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            **kwargs
        )
        
        return self.history
    
    def _get_callbacks(self, patience: int) -> list:
        """Create training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=str(self.model_dir / f"model_{timestamp}_{{epoch:02d}}.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            TensorBoard(
                log_dir=str(self.log_dir / timestamp),
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def save_model(self, filepath: Path):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_history(self) -> Optional[Dict[str, Any]]:
        """Get training history."""
        if self.history is None:
            return None
        return self.history.history
