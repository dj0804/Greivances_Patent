"""
API routes for grievance prediction.
"""

from fastapi import APIRouter, HTTPException
from typing import List
import time
import asyncio
import os
import logging
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
from scripts.database_manager import HostelDB

try:
    from decision_engine.engine.orchestrator import DecisionEngineOrchestrator
    from decision_engine.services.data_interface import Complaint as EngineComplaint
except ImportError:
    DecisionEngineOrchestrator = None
    EngineComplaint = None
    logger_import = __import__("logging").getLogger(__name__)
    logger_import.warning(
        "New DecisionEngineOrchestrator could not be imported; "
        "cluster-level processing will be skipped."
    )


router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize predictor (in production, load this once at startup)
# This is a placeholder - update with actual model path
predictor = None
decision_engine = UrgencyDecisionEngine()
recent_predictions = deque(maxlen=200)

try:
    orchestrator = DecisionEngineOrchestrator() if DecisionEngineOrchestrator is not None else None
except Exception as _orch_exc:
    orchestrator = None
    logging.getLogger(__name__).warning(
        "Failed to initialize DecisionEngineOrchestrator: %s", _orch_exc
    )

vector_store = None

# Env toggle for safe rollout and temporary fallback mode.
DB_PERSISTENCE_ENABLED = os.getenv("ENABLE_DB_PERSISTENCE", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
db_manager = None

if DB_PERSISTENCE_ENABLED:
    try:
        db_manager = HostelDB()
        db_manager.initialize_tables()
        logger.info("Database persistence initialized")
    except Exception as exc:
        db_manager = None
        logger.exception(
            "Database initialization failed; continuing without DB writes: %s",
            exc,
        )
else:
    logger.info("Database persistence disabled via ENABLE_DB_PERSISTENCE")


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


def _store_prediction(request: ComplaintRequest, result: dict) -> dict:
    decision_details = result.get("decision_details") or {}
    created_at = _to_iso(request.timestamp)
    score = float(result.get("confidence", 0.0))
    urgency = str(result.get("urgency_level", "Unknown"))
    sla_hours = int(decision_details.get("sla_hours", 24))
    sla_breach = (_to_datetime(created_at) + timedelta(hours=sla_hours)) < datetime.now(timezone.utc)

    record = {
        "id": _next_complaint_id(),
        "text": request.complaint_text,
        "summary": result.get("summary"),
        "urgency": urgency,
        "score": score,
        "cluster": decision_details.get("cluster_id"),
        "created_at": created_at,
        "sla_hours": sla_hours,
        "sla_breach": sla_breach,
    }

    recent_predictions.appendleft(record)
    return record


def _build_db_incident_payload(request: ComplaintRequest, result: dict, record: dict) -> dict:
    decision_details = result.get("decision_details") or {}
    return {
        "complaint_id": record["id"],
        "timestamp": _to_datetime(record["created_at"]),
        "raw_text": request.complaint_text,
        "preprocessing_data": {
            "hostel_id": request.hostel_id,
            "complaint_type": request.complaint_type,
            "urgency_level": result.get("urgency_level"),
            "confidence": float(result.get("confidence", 0.0)),
            "summary": result.get("summary"),
            "decision_details": decision_details,
            "cluster_id": decision_details.get("cluster_id"),
            "sla_hours": int(decision_details.get("sla_hours", 24)),
            "sla_breach": bool(record.get("sla_breach")),
        },
    }


def _persist_single_to_db(request: ComplaintRequest, result: dict, record: dict) -> None:
    if not DB_PERSISTENCE_ENABLED or db_manager is None:
        return

    incident = _build_db_incident_payload(request, result, record)
    try:
        db_manager.insert_incident(
            complaint_id=incident["complaint_id"],
            timestamp=incident["timestamp"],
            raw_text=incident["raw_text"],
            preprocessing_data=incident["preprocessing_data"],
        )
        logger.info("Complaint persisted to DB")
    except Exception as exc:
        logger.exception("Failed to persist complaint to DB: %s", exc)


def _persist_batch_to_db(incidents: List[dict]) -> None:
    if not incidents or not DB_PERSISTENCE_ENABLED or db_manager is None:
        return

    try:
        db_manager.insert_incidents_bulk(incidents)
        logger.info("Complaint persisted to DB")
    except Exception as exc:
        logger.exception("Failed to persist complaint batch to DB: %s", exc)


def load_vector_store():
    """Lazily load the FAISS vector store from disk."""
    global vector_store
    if vector_store is not None:
        return vector_store
    import pickle as _pickle
    vs_path = MODELS_DIR / "vector_store.pkl"
    if not vs_path.exists():
        logger.warning("Vector store not found at %s; cluster assignment disabled.", vs_path)
        return None
    try:
        with open(vs_path, "rb") as f:
            vector_store = _pickle.load(f)
    except Exception as exc:
        logger.warning("Failed to load vector store: %s", exc)
    return vector_store


def build_engine_complaint(
    complaint_id: str,
    text: str,
    urgency_score: float,
    timestamp,
    cluster_id: str = "default",
) -> "EngineComplaint":
    """Construct an EngineComplaint for the new orchestrator."""
    from datetime import datetime, timezone
    ts = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
    return EngineComplaint(
        complaint_id=complaint_id,
        cluster_id=cluster_id,
        urgency_score=max(0.0, min(1.0, float(urgency_score))),
        timeline_score=0.5,
        timestamp=ts,
        text=text,
    )


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
        
        tokenizer_pkl = MODELS_DIR / "tokenizer.pkl"
        predictor = GrievancePredictor(
            model_path=model_path,
            fasttext_model_path=fasttext_path,
            tokenizer_path=tokenizer_pkl if tokenizer_pkl.exists() else None,
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
        
        # Online cluster assignment via FAISS vector store
        assigned_cluster_id = None
        try:
            vs = load_vector_store()
            if vs is not None:
                emb = vs.get_embedding_for_text(request.complaint_text)
                if emb is not None:
                    assigned_cluster_id = vs.assign_cluster(emb)
        except Exception as _vs_err:
            logger.warning("Cluster assignment failed: %s", _vs_err)
        result["cluster_id"] = assigned_cluster_id

        # Apply decision engine
        decision = decision_engine.make_decision(
            model_prediction=result,
            temporal_features=result.get('temporal_features')
        )
        
        result['decision_details'] = decision
        result['urgency_level'] = decision['urgency_level']
        result['confidence'] = decision['confidence']
        result['summary'] = summary

        # Augment with new orchestrator breakdown (additive — does not replace existing details)
        if orchestrator is not None and EngineComplaint is not None:
            try:
                cluster_id = assigned_cluster_id or result.get("cluster_id") or "singleton"
                engine_complaint = build_engine_complaint(
                    complaint_id=_next_complaint_id(),
                    text=request.complaint_text,
                    urgency_score=result["confidence"],
                    timestamp=request.timestamp,
                    cluster_id=cluster_id,
                )
                orch_results = orchestrator.process_complaints([engine_complaint])
                if orch_results:
                    top = orch_results[0]
                    result["decision_details"]["orchestrator_breakdown"] = top.breakdown.model_dump()
            except Exception as _orch_err:
                logger.warning("Orchestrator processing failed: %s", _orch_err)

        stored_record = _store_prediction(request, result)
        _persist_single_to_db(request, result, stored_record)
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            data=UrgencyPrediction(**result),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_single_complaint(complaint: ComplaintRequest, pred, loop) -> tuple:
    """
    Process one complaint end-to-end and return (UrgencyPrediction, db_incident_dict).
    Each call operates on its own local variables — no shared mutable state.
    """
    summary_future = loop.run_in_executor(
        None, summarizer_instance.summarize, complaint.complaint_text
    )

    result = pred.predict(
        complaint_text=complaint.complaint_text,
        timestamp=complaint.timestamp,
        hostel_id=complaint.hostel_id,
        complaint_type=complaint.complaint_type,
        return_probabilities=True,
    )

    summary = await summary_future

    decision = decision_engine.make_decision(
        model_prediction=result,
        temporal_features=result.get("temporal_features"),
    )

    result["decision_details"] = decision
    result["urgency_level"] = decision["urgency_level"]
    result["confidence"] = decision["confidence"]
    result["summary"] = summary

    stored_record = _store_prediction(complaint, result)
    db_incident = _build_db_incident_payload(complaint, result, stored_record)

    return UrgencyPrediction(**result), db_incident


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict urgency for multiple complaints"
)
async def predict_batch(request: ComplaintBatchRequest):
    """
    Predict urgency for multiple complaints in batch (parallelised).
    """
    try:
        start_time = time.time()

        pred = get_predictor()
        loop = asyncio.get_event_loop()

        tasks = [process_single_complaint(c, pred, loop) for c in request.complaints]
        task_results = await asyncio.gather(*tasks)

        results = [r[0] for r in task_results]
        batch_incidents = [r[1] for r in task_results]

        _persist_batch_to_db(batch_incidents)

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            success=True,
            data=results,
            count=len(results),
            processing_time_ms=processing_time,
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


def _get_all_predictions_for_dashboard() -> list:
    """Return all predictions for dashboard analytics, preferring DB over in-memory deque."""
    if db_manager is not None:
        try:
            rows = db_manager.get_recent_incidents(limit=200)
            if rows:
                return rows
        except Exception as exc:
            logger.warning("DB dashboard fetch failed, falling back to deque: %s", exc)
    return list(recent_predictions)


@router.get(
    "/dashboard",
    tags=["Information"],
    summary="Get dashboard-ready analytics and recent complaints"
)
async def get_dashboard_data():
    all_predictions = _get_all_predictions_for_dashboard()
    urgency_counter = Counter(item["urgency"].lower() for item in all_predictions)
    total_active = len(all_predictions)
    critical = urgency_counter.get("high", 0)
    sla_overdue = sum(1 for item in all_predictions if item.get("sla_breach"))

    cluster_counter = Counter(str(item.get("cluster") or "N/A") for item in all_predictions)
    cluster_distribution = [
        {"name": f"Cluster {cluster}", "count": count}
        for cluster, count in cluster_counter.most_common(6)
    ]

    today = datetime.now(timezone.utc).date()
    trend = []
    for days_ago in range(4, -1, -1):
        day = today - timedelta(days=days_ago)
        day_items = [
            item for item in all_predictions
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
    for item in all_predictions[:20]:
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
