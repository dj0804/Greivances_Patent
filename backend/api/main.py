"""
FastAPI application for grievance urgency prediction.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .schemas import HealthResponse


# Initialize FastAPI app
app = FastAPI(
    title="Hostel Grievance Urgency API",
    description="API for predicting urgency of hostel complaints",
    version="0.1.0"
)

# ALLOWED_ORIGINS controls which origins may call the API.
# Set it to a comma-separated list of origins, e.g.:
#   ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
# Defaults to http://localhost:3000 for local development.
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Hostel Grievance Urgency API", "status": "running"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")

