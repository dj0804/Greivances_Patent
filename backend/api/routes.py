"""
API routes for grievance prediction.
"""

from fastapi import APIRouter, HTTPException
from typing import List
import time
import asyncio
from pathlib import Path
from collections import Counter, deque
from datetime import datetime, timedelta, timezone

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
from src.summarization.summarizer import summarizer_instance


router = APIRouter()

# Initialize predictor (in production, load this once at startup)
# This is a placeholder - update with actual model path
predictor = None
decision_engine = UrgencyDecisionEngine()
recent_predictions = deque(maxlen=200)


def _next_complaint_id() -> str:
    return f"C-{int(time.time() * 1000)}"


def _to_iso(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    return datetime.now(timezone.utc).isoformat()


def _to_datetime(value: str) -> datetime:
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except Exception:
        return datetime.now(timezone.utc)


def _format_age(created_at: str) -> str:
    now = datetime.now(timezone.utc)
    created = _to_datetime(created_at)
    delta = now - created
    hours = int(delta.total_seconds() // 3600)
    if hours <= 0:
        minutes = max(1, int(delta.total_seconds() // 60))
        return f"{minutes}m ago"
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def _store_prediction(request: ComplaintRequest, result: dict):
    decision_details = result.get("decision_details") or {}
    created_at = _to_iso(request.timestamp)
    score = float(result.get("confidence", 0.0))
    urgency = str(result.get("urgency_level", "Unknown"))
    sla_hours = int(decision_details.get("sla_hours", 24))
    sla_breach = (_to_datetime(created_at) + timedelta(hours=sla_hours)) < datetime.now(timezone.utc)

    recent_predictions.appendleft({
        "id": _next_complaint_id(),
        "text": request.complaint_text,
        "summary": result.get("summary"),
        "urgency": urgency,
        "score": score,
        "cluster": decision_details.get("cluster_id"),
        "created_at": created_at,
        "sla_hours": sla_hours,
        "sla_breach": sla_breach,
    })


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
        
        # Only load FastText if the file actually exists
        fasttext_path = None
        if FASTTEXT_MODEL_PATH:
            ft_path = Path(FASTTEXT_MODEL_PATH)
            if ft_path.exists():
                fasttext_path = ft_path
            else:
                # FastText is optional - log a warning but continue without it
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"FastText model not found at {FASTTEXT_MODEL_PATH}. Continuing without semantic embeddings.")
        
        predictor = GrievancePredictor(
            model_path=model_path,
            fasttext_model_path=fasttext_path
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
        
        # Prepare tasks for parallel execution
        loop = asyncio.get_event_loop()
        
        # Summary generation runs in threadpool to not block the event loop
        summary_future = loop.run_in_executor(None, summarizer_instance.summarize, request.complaint_text)
        
        # Make prediction (synchronous operation)
        result = pred.predict(
            complaint_text=request.complaint_text,
            timestamp=request.timestamp,
            hostel_id=request.hostel_id,
            complaint_type=request.complaint_type,
            return_probabilities=True
        )
        
        # Wait for summary to finish
        summary = await summary_future
        
        # Apply decision engine
        decision = decision_engine.make_decision(
            model_prediction=result,
            temporal_features=result.get('temporal_features')
        )
        
        result['decision_details'] = decision
        result['urgency_level'] = decision['urgency_level']
        result['confidence'] = decision['confidence']
        result['summary'] = summary
        _store_prediction(request, result)
        
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
            # Prepare tasks for parallel execution
            loop = asyncio.get_event_loop()
            summary_future = loop.run_in_executor(None, summarizer_instance.summarize, complaint.complaint_text)
            
            result = pred.predict(
                complaint_text=complaint.complaint_text,
                timestamp=complaint.timestamp,
                hostel_id=complaint.hostel_id,
                complaint_type=complaint.complaint_type,
                return_probabilities=True
            )
            
            summary = await summary_future
            
            # Apply decision engine
            decision = decision_engine.make_decision(
                model_prediction=result,
                temporal_features=result.get('temporal_features')
            )
            
            result['decision_details'] = decision
            result['urgency_level'] = decision['urgency_level']
            result['confidence'] = decision['confidence']
            result['summary'] = summary
            _store_prediction(complaint, result)
            
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


@router.get(
    "/dashboard",
    tags=["Information"],
    summary="Get dashboard-ready analytics and recent complaints"
)
async def get_dashboard_data():
    urgency_counter = Counter(item["urgency"].lower() for item in recent_predictions)
    total_active = len(recent_predictions)
    critical = urgency_counter.get("high", 0)
    sla_overdue = sum(1 for item in recent_predictions if item.get("sla_breach"))

    cluster_counter = Counter(str(item.get("cluster") or "N/A") for item in recent_predictions)
    cluster_distribution = [
        {"name": f"Cluster {cluster}", "count": count}
        for cluster, count in cluster_counter.most_common(6)
    ]

    today = datetime.now(timezone.utc).date()
    trend = []
    for days_ago in range(4, -1, -1):
        day = today - timedelta(days=days_ago)
        day_items = [
            item for item in recent_predictions
            if _to_datetime(item["created_at"]).date() == day
        ]
        day_counter = Counter(item["urgency"].lower() for item in day_items)
        trend.append({
            "name": day.strftime("%a"),
            "high": day_counter.get("high", 0),
            "medium": day_counter.get("medium", 0),
            "low": day_counter.get("low", 0),
        })

    recent = []
    for item in list(recent_predictions)[:20]:
        recent.append({
            "id": item["id"],
            "text": item.get("text"),
            "summary": item.get("summary"),
            "urgency": item.get("urgency"),
            "score": item.get("score", 0.0),
            "cluster": item.get("cluster"),
            "time": _format_age(item["created_at"]),
            "slaBreach": bool(item.get("sla_breach")),
        })

    return {
        "stats": {
            "total_active": total_active,
            "critical_urgency": critical,
            "active_clusters": len(cluster_counter),
            "sla_overdue": sla_overdue,
        },
        "trend": trend,
        "clusters": cluster_distribution,
        "recent_complaints": recent,
    }
