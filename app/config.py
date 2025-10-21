"""Configuration settings for the adaptive learning assessment system."""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "Adaptive Learning Assessment System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./adaptive_learning.db"
    redis_url: str = "redis://localhost:6379"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # ML Models
    model_storage_path: str = "data/models"
    training_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Assessment Settings
    max_questions_per_assessment: int = 50
    min_questions_per_assessment: int = 10
    assessment_time_limit_minutes: int = 60
    
    # IRT Model Settings
    irt_difficulty_range: tuple = (-3.0, 3.0)
    irt_discrimination_range: tuple = (0.1, 2.0)
    irt_guessing_range: tuple = (0.0, 0.3)
    
    # BKT Model Settings
    bkt_initial_knowledge: float = 0.1
    bkt_learn_rate: float = 0.3
    bkt_guess_rate: float = 0.1
    bkt_slip_rate: float = 0.1
    
    # XGBoost Settings
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # API Settings
    api_v1_prefix: str = "/api/v1"
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
