"""
Unit tests for inference pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from datetime import datetime
from src.decision_engine import UrgencyDecisionEngine


class TestDecisionEngine(unittest.TestCase):
    """Test cases for urgency decision engine."""
    
    def setUp(self):
        self.engine = UrgencyDecisionEngine()
    
    def test_basic_decision(self):
        """Test basic decision making."""
        model_prediction = {
            'urgency_level': 'Medium',
            'confidence': 0.6,
            'cleaned_text': 'water heater broken'
        }
        
        decision = self.engine.make_decision(model_prediction)
        
        self.assertIn('urgency_level', decision)
        self.assertIn('confidence', decision)
        self.assertIn('adjustments', decision)
    
    def test_temporal_boost(self):
        """Test temporal context boost."""
        model_prediction = {
            'urgency_level': 'Medium',
            'confidence': 0.6,
            'cleaned_text': 'water heater broken'
        }
        
        temporal_features = {
            'temporal': {
                'is_night': True,
                'is_peak_hour': False,
                'is_weekend': False
            }
        }
        
        decision = self.engine.make_decision(
            model_prediction,
            temporal_features=temporal_features
        )
        
        # Confidence should be boosted
        self.assertGreater(decision['confidence'], model_prediction['confidence'])
        self.assertTrue(len(decision['adjustments']) > 0)
    
    def test_keyword_detection(self):
        """Test critical keyword detection."""
        model_prediction = {
            'urgency_level': 'Low',
            'confidence': 0.4,
            'cleaned_text': 'emergency broken pipe flooding'
        }
        
        decision = self.engine.make_decision(model_prediction)
        
        # Should detect emergency keywords
        keyword_adjustment = [
            adj for adj in decision['adjustments']
            if adj['type'] == 'keyword'
        ]
        self.assertTrue(len(keyword_adjustment) > 0)
    
    def test_history_boost(self):
        """Test repeated complaint boost."""
        model_prediction = {
            'urgency_level': 'Low',
            'confidence': 0.4,
            'cleaned_text': 'water issue'
        }
        
        complaint_history = [
            {'timestamp': '2026-02-10', 'issue': 'water'},
            {'timestamp': '2026-02-15', 'issue': 'water'}
        ]
        
        decision = self.engine.make_decision(
            model_prediction,
            complaint_history=complaint_history
        )
        
        # Should boost due to repeated complaints
        self.assertGreater(decision['confidence'], model_prediction['confidence'])


if __name__ == '__main__':
    unittest.main()
