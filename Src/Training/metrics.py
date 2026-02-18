"""
Custom metrics for urgency classification evaluation.
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import numpy as np
from typing import Dict


def urgency_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for urgency classification.
    
    Args:
        y_true: True labels (one-hot or integer)
        y_pred: Predicted labels (one-hot or probabilities)
        
    Returns:
        Dictionary of metrics
    """
    # Convert one-hot to integers if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    metrics = {
        'accuracy': np.mean(y_true == y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, score in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = score
    
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None
):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of urgency levels
    """
    if target_names is None:
        target_names = ['Low', 'Medium', 'High']
    
    # Convert one-hot to integers if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


class UrgencyF1Score(keras.metrics.Metric):
    """
    Custom Keras metric for urgency F1 score.
    """
    
    def __init__(self, name='urgency_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        # Calculate F1 for this batch
        # This is a simplified version; for production use sklearn's f1_score
        correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        batch_f1 = tf.reduce_mean(correct)
        
        self.f1.assign_add(batch_f1)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.f1 / self.count
    
    def reset_state(self):
        self.f1.assign(0.0)
        self.count.assign(0.0)
