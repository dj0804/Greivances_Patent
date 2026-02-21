"""
FastAPI Main Application - Decision Engine

This module provides REST API endpoints for the Decision Engine,
enabling external systems to submit complaints and retrieve urgency rankings.

Endpoints:
---------
POST /decision/process
    - Process complaints and compute cluster urgencies
    - Returns ranked clusters

GET /decision/ranked
    - Retrieve last computed rankings
    - Optional filtering by top-n or threshold

GET /health
    - Health check endpoint

Design:
-------
- Stateless: Each request is independent
- Thread-safe: Uses immutable config and stateless components
- RESTful: Follows REST conventions
- Documented: OpenAPI/Swagger auto-generation
- Error handling: Comprehensive error responses
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import List, Optional
import logging

from decision_engine.config import DecisionEngineConfig
from decision_engine.services.data_interface import Complaint, InMemoryComplaintDataSource
from decision_engine.engine.orchestrator import DecisionEngineOrchestrator
from decision_engine.schemas.request_models import (
    ProcessDecisionRequest,
    GetRankedRequest,
    ComplaintInput
)
from decision_engine.schemas.response_models import (
    ProcessDecisionResponse,
    GetRankedResponse,
    ClusterSummary,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Decision Engine API",
    description="Complaint intelligence system for computing cluster-level urgency",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state (in production, use database or cache)
_last_processed_results: Optional[List[ClusterSummary]] = None
_last_processed_time: Optional[datetime] = None

# Initialize orchestrator
_config = DecisionEngineConfig()
_orchestrator = DecisionEngineOrchestrator(_config)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status information about the service
    """
    return {
        "status": "healthy",
        "service": "decision-engine",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/decision/process",
    response_model=ProcessDecisionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Decision Engine"]
)
async def process_decision(request: ProcessDecisionRequest) -> ProcessDecisionResponse:
    """
    Process complaints and compute cluster urgencies.
    
    This endpoint:
    1. Accepts a list of complaints with cluster assignments
    2. Groups by cluster_id
    3. Computes urgency for each cluster
    4. Returns ranked list of clusters
    
    The clustering is performed externally (FAISS, VectorDB, etc.)
    and cluster_id is provided in the request.
    
    Args:
        request: ProcessDecisionRequest containing complaints
        
    Returns:
        ProcessDecisionResponse with ranked clusters
        
    Raises:
        400: Invalid input data
        500: Internal processing error
    """
    global _last_processed_results, _last_processed_time
    
    try:
        logger.info(f"Received process request with {len(request.complaints)} complaints")
        
        # Convert Pydantic models to domain objects
        complaints = [
            Complaint(
                complaint_id=c.complaint_id,
                cluster_id=c.cluster_id,
                urgency_score=c.urgency_score,
                timeline_score=c.timeline_score,
                timestamp=c.timestamp,
                text=c.text
            )
            for c in request.complaints
        ]
        
        # Process through orchestrator
        results = _orchestrator.process_complaints(complaints)
        
        # Cache results
        _last_processed_results = results
        _last_processed_time = datetime.now()
        
        # Count unique clusters
        unique_clusters = len(set(c.complaint_id for c in request.complaints))
        
        logger.info(f"Successfully processed {len(results)} clusters")
        
        return ProcessDecisionResponse(
            status="success",
            clusters_processed=len(results),
            total_complaints=len(request.complaints),
            ranked_clusters=results,
            processing_timestamp=_last_processed_time
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing error: {str(e)}"
        )


@app.get(
    "/decision/ranked",
    response_model=GetRankedResponse,
    status_code=status.HTTP_200_OK,
    tags=["Decision Engine"]
)
async def get_ranked_clusters(
    top_n: Optional[int] = None,
    min_urgency_threshold: Optional[float] = None
) -> GetRankedResponse:
    """
    Retrieve last computed cluster rankings.
    
    This endpoint returns the results from the most recent call to /decision/process.
    
    Optional filtering:
    - top_n: Return only top N clusters
    - min_urgency_threshold: Filter clusters below threshold
    
    Args:
        top_n: Maximum number of clusters to return
        min_urgency_threshold: Minimum urgency score
        
    Returns:
        GetRankedResponse with filtered clusters
        
    Raises:
        404: No processed results available
        400: Invalid filter parameters
    """
    global _last_processed_results, _last_processed_time
    
    if _last_processed_results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No processed results available. Call /decision/process first."
        )
    
    try:
        # Apply filters
        filtered_results = _last_processed_results
        
        # Filter by urgency threshold
        if min_urgency_threshold is not None:
            if not (0.0 <= min_urgency_threshold <= 1.0):
                raise ValueError("min_urgency_threshold must be in [0, 1]")
            filtered_results = [
                r for r in filtered_results 
                if r.final_urgency >= min_urgency_threshold
            ]
        
        # Limit to top N
        if top_n is not None:
            if top_n < 1:
                raise ValueError("top_n must be >= 1")
            filtered_results = filtered_results[:top_n]
        
        logger.info(
            f"Returning {len(filtered_results)} clusters "
            f"(top_n={top_n}, threshold={min_urgency_threshold})"
        )
        
        return GetRankedResponse(
            clusters=filtered_results,
            total_clusters=len(_last_processed_results),
            retrieved_at=datetime.now()
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.delete(
    "/decision/cache",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Decision Engine"]
)
async def clear_cache():
    """
    Clear cached results and calibration memory.
    
    Useful for:
    - Testing
    - Memory management
    - Reset state
    """
    global _last_processed_results, _last_processed_time
    
    _last_processed_results = None
    _last_processed_time = None
    _orchestrator.clear_calibration_memory()
    
    logger.info("Cleared all cache and memory")
    
    return None


@app.get(
    "/decision/config",
    tags=["Configuration"]
)
async def get_configuration():
    """
    Get current engine configuration.
    
    Returns:
        Current configuration parameters
    """
    return {
        "aggregation": {
            "alpha": _config.aggregation.alpha,
            "beta": _config.aggregation.beta,
            "delta": _config.aggregation.delta,
            "percentile_threshold": _config.aggregation.percentile_threshold
        },
        "temporal": {
            "theta_1": _config.temporal.theta_1,
            "theta_2": _config.temporal.theta_2,
            "recent_window_hours": _config.temporal.recent_window_hours,
            "historical_window_hours": _config.temporal.historical_window_hours
        },
        "calibration": {
            "gamma": _config.calibration.gamma,
            "enable_smoothing": _config.calibration.enable_smoothing
        },
        "fusion": {
            "lambda_1": _config.fusion.lambda_1,
            "lambda_2": _config.fusion.lambda_2
        }
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
