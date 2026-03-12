"""
Urgency decision logic combining model predictions with temporal context.
"""

from typing import Dict, Any
from datetime import datetime


class UrgencyDecisionEngine:
    """
    Combines ML model output with rule-based temporal logic.
    """
    
    def __init__(
        self,
        urgency_threshold: float = 0.7,
        escalation_rules: Dict[str, Any] = None
    ):
        """
        Initialize decision engine.
        
        Args:
            urgency_threshold: Confidence threshold for urgency escalation
            escalation_rules: Custom escalation rules
        """
        self.urgency_threshold = urgency_threshold
        self.escalation_rules = escalation_rules or self._default_escalation_rules()
    
    def _default_escalation_rules(self) -> Dict[str, Any]:
        """Define default escalation rules."""
        return {
            'peak_hour_boost': 0.1,  # Boost urgency during peak hours
            'night_boost': 0.15,      # Higher boost for night complaints
            'weekend_boost': 0.05,    # Weekend boost
            'repeated_complaint_boost': 0.2,  # Boost for repeated issues
            'critical_keywords': [
                'emergency', 'urgent', 'immediately', 'asap',
                'broken', 'flooded', 'injured', 'fire'
            ],
        }
    
    def make_decision(
        self,
        model_prediction: Dict[str, Any],
        temporal_features: Dict[str, Any] = None,
        complaint_history: list = None
    ) -> Dict[str, Any]:
        """
        Make final urgency decision combining model and rules.
        
        Args:
            model_prediction: Output from ML model
            temporal_features: Temporal context features
            complaint_history: Previous complaints from same source
            
        Returns:
            Final decision with urgency level and reasoning
        """
        base_urgency = model_prediction['urgency_level']
        base_confidence = model_prediction['confidence']
        
        # Calculate adjustments
        adjustments = []
        boost = 0.0
        
        # Temporal adjustments
        if temporal_features:
            temporal_boost = self._calculate_temporal_boost(temporal_features)
            boost += temporal_boost
            if temporal_boost > 0:
                adjustments.append({
                    'type': 'temporal',
                    'boost': temporal_boost,
                    'reason': self._get_temporal_reason(temporal_features)
                })
        
        # Keyword-based adjustments
        text = model_prediction.get('cleaned_text', '')
        keyword_boost = self._check_critical_keywords(text)
        if keyword_boost > 0:
            boost += keyword_boost
            adjustments.append({
                'type': 'keyword',
                'boost': keyword_boost,
                'reason': 'Critical keywords detected'
            })
        
        # History-based adjustments
        if complaint_history and len(complaint_history) > 0:
            history_boost = self.escalation_rules['repeated_complaint_boost']
            boost += history_boost
            adjustments.append({
                'type': 'history',
                'boost': history_boost,
                'reason': f'Repeated complaint (count: {len(complaint_history)})'
            })
        
        # Calculate final urgency
        adjusted_confidence = min(1.0, base_confidence + boost)
        
        # Determine final urgency level
        final_urgency = self._determine_final_urgency(
            base_urgency,
            adjusted_confidence
        )
        
        return {
            'urgency_level': final_urgency,
            'confidence': adjusted_confidence,
            'original_urgency': base_urgency,
            'original_confidence': base_confidence,
            'adjustments': adjustments,
            'total_boost': boost,
            'requires_immediate_action': adjusted_confidence >= self.urgency_threshold,
        }
    
    def _calculate_temporal_boost(self, temporal_features: Dict[str, Any]) -> float:
        """Calculate urgency boost based on temporal context."""
        boost = 0.0
        temporal = temporal_features.get('temporal', {})
        
        if temporal.get('is_peak_hour'):
            boost += self.escalation_rules['peak_hour_boost']
        
        if temporal.get('is_night'):
            boost += self.escalation_rules['night_boost']
        
        if temporal.get('is_weekend'):
            boost += self.escalation_rules['weekend_boost']
        
        return boost
    
    def _get_temporal_reason(self, temporal_features: Dict[str, Any]) -> str:
        """Generate explanation for temporal boost."""
        temporal = temporal_features.get('temporal', {})
        reasons = []
        
        if temporal.get('is_night'):
            reasons.append('night-time')
        if temporal.get('is_peak_hour'):
            reasons.append('peak hours')
        if temporal.get('is_weekend'):
            reasons.append('weekend')
        
        return f"Complaint filed during {', '.join(reasons)}"
    
    def _check_critical_keywords(self, text: str) -> float:
        """Check for critical urgency keywords."""
        text_lower = text.lower()
        keywords_found = [
            kw for kw in self.escalation_rules['critical_keywords']
            if kw in text_lower
        ]
        
        if keywords_found:
            # Boost proportional to number of critical keywords
            return min(0.3, len(keywords_found) * 0.1)
        
        return 0.0
    
    def _determine_final_urgency(
        self,
        base_urgency: str,
        adjusted_confidence: float
    ) -> str:
        """Determine final urgency level."""
        # Map confidence to urgency levels
        if adjusted_confidence >= 0.8:
            return 'High'
        elif adjusted_confidence >= 0.5:
            return 'Medium'
        else:
            # Don't downgrade if base urgency was already high
            if base_urgency == 'High':
                return 'Medium'
            return 'Low'
