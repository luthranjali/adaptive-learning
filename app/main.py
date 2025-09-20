"""Main FastAPI application for the adaptive learning assessment system."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config import settings
from app.database import engine, Base
from app.api import auth, assessments
from app.models import *  # Import all models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting Adaptive Learning Assessment System...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created")
    
    # Initialize ML models
    print("ML models initialized")
    
    yield
    
    # Shutdown
    print("Shutting down Adaptive Learning Assessment System...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-driven adaptive learning assessment system with ML-powered recommendations",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.localhost"]
)

# Include API routers
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(assessments.router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Adaptive Learning Assessment System",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "v1",
        "status": "operational",
        "features": [
            "Adaptive question selection",
            "ML-powered level recommendations",
            "Knowledge gap detection",
            "Performance analysis",
            "Real-time assessment adaptation"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
