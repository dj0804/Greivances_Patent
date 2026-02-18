"""
API routes for grievance prediction.
"""

from fastapi import APIRouter, HTTPException
from typing import List
import time
from pathlib import Path

from .schemas import (
    ComplaintRequest,
    ComplaintBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
    UrgencyPrediction,
    ErrorResponse
)
from src.inference import GrievancePredictor
from src.decision_engine import UrgencyDecisionEngine
from src.config import MODELS_DIR, FASTTEXT_MODEL_PATH


router = APIRouter()

# Initialize predictor (in production, load this once at startup)
# This is a placeholder - update with actual model path
predictor = None
decision_engine = UrgencyDecisionEngine()


def get_predictor():
    """Lazy load predictor."""
    global predictor
    if predictor is None:
        # Update with actual model path
        model_path = MODELS_DIR / "best_model.h5"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Please train a model first."
            )
        
        predictor = GrievancePredictor(
            model_path=model_path,
            fasttext_model_path=Path(FASTTEXT_MODEL_PATH) if FASTTEXT_MODEL_PATH else None
        )
    return predictor


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict urgency for a single complaint"
)
async def predict_urgency(request: ComplaintRequest):
    """
    Predict urgency level for a single complaint.
    
    Returns urgency classification with confidence score and decision details.
    """
    try:
        start_time = time.time()
        
        # Get predictor
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict(
            complaint_text=request.complaint_text,
            timestamp=request.timestamp,
            hostel_id=request.hostel_id,
            complaint_type=request.complaint_type,
            return_probabilities=True
        )
        
        # Apply decision engine
        decision = decision_engine.make_decision(
            model_prediction=result,
            temporal_features=result.get('temporal_features')
        )
        
        result['decision_details'] = decision
        result['urgency_level'] = decision['urgency_level']
        result['confidence'] = decision['confidence']
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            data=UrgencyPrediction(**result),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict urgency for multiple complaints"
)
async def predict_batch(request: ComplaintBatchRequest):
    """
    Predict urgency for multiple complaints in batch.
    """
    try:
        start_time = time.time()
        
        # Get predictor
        pred = get_predictor()
        
        results = []
        for complaint in request.complaints:
            result = pred.predict(
                complaint_text=complaint.complaint_text,
                timestamp=complaint.timestamp,
                hostel_id=complaint.hostel_id,
                complaint_type=complaint.complaint_type,
                return_probabilities=True
            )
            
            # Apply decision engine
            decision = decision_engine.make_decision(
                model_prediction=result,
                temporal_features=result.get('temporal_features')
            )
            
            result['decision_details'] = decision
            result['urgency_level'] = decision['urgency_level']
            result['confidence'] = decision['confidence']
            
            results.append(UrgencyPrediction(**result))
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            success=True,
            data=results,
            count=len(results),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    tags=["Information"],
    summary="Get model statistics"
)
async def get_stats():
    """Get model and system statistics."""
    return {
        "model_loaded": predictor is not None,
        "urgency_levels": ["Low", "Medium", "High"],
        "features": [
            "Text cleaning and tokenization",
            "Temporal context encoding",
            "FastText embeddings",
            "CNN-BiLSTM architecture",
            "Rule-based decision engine"
        ]
    }
