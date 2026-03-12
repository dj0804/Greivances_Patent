"""
Custom loss functions for urgency classification.
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def weighted_categorical_crossentropy(class_weights: dict):
    """
    Create weighted categorical crossentropy loss.
    
    Useful for handling class imbalance in urgency levels.
    
    Args:
        class_weights: Dictionary mapping class indices to weights
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        # Convert class weights to tensor
        weights = tf.constant([
            class_weights.get(i, 1.0) for i in range(len(class_weights))
        ])
        
        # Apply weights
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.gather(weights, tf.argmax(y_true, axis=1))
        
        # Calculate categorical crossentropy
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Apply weights
        weighted_ce = ce * weights
        
        return K.mean(weighted_ce)
    
    return loss


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal loss for handling class imbalance.
    
    Focuses training on hard examples.
    
    Args:
        gamma: Focusing parameter
        alpha: Weighting factor
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        ce = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        fl = weight * ce
        
        return K.mean(K.sum(fl, axis=1))
    
    return loss
