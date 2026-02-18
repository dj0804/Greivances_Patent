"""
Unit tests for model architectures.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
from src.models import CNNModel, BiLSTMModel, CNNBiLSTMModel, DenseHead


class TestModels(unittest.TestCase):
    """Test cases for neural network models."""
    
    def setUp(self):
        self.vocab_size = 1000
        self.embedding_dim = 100
        self.max_length = 50
        self.num_classes = 3
    
    def test_cnn_model_creation(self):
        """Test CNN model can be created."""
        model = CNNModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            num_classes=self.num_classes
        )
        
        self.assertIsNotNone(model.model)
        model.compile()
        
        # Test model output shape
        dummy_input = np.random.randint(0, self.vocab_size, (1, self.max_length))
        output = model.model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (1, self.num_classes))
    
    def test_bilstm_model_creation(self):
        """Test BiLSTM model can be created."""
        model = BiLSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            num_classes=self.num_classes
        )
        
        self.assertIsNotNone(model.model)
        model.compile()
        
        # Test model output shape
        dummy_input = np.random.randint(0, self.vocab_size, (1, self.max_length))
        output = model.model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (1, self.num_classes))
    
    def test_cnn_bilstm_model_creation(self):
        """Test combined CNN-BiLSTM model can be created."""
        model = CNNBiLSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            num_classes=self.num_classes
        )
        
        self.assertIsNotNone(model.model)
        model.compile()
        
        # Test model output shape
        dummy_input = np.random.randint(0, self.vocab_size, (1, self.max_length))
        output = model.model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (1, self.num_classes))
    
    def test_dense_head_creation(self):
        """Test dense head model can be created."""
        model = DenseHead(
            input_dim=128,
            temporal_dim=10,
            num_classes=self.num_classes
        )
        
        self.assertIsNotNone(model.model)
        model.compile()
        
        # Test model with dual inputs
        text_features = np.random.rand(1, 128)
        temporal_features = np.random.rand(1, 10)
        output = model.model.predict(
            [text_features, temporal_features],
            verbose=0
        )
        self.assertEqual(output.shape, (1, self.num_classes))


if __name__ == '__main__':
    unittest.main()
