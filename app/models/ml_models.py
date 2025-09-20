"""ML model tracking and versioning."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON
from sqlalchemy.sql import func
from app.database import Base


class MLModel(Base):
    """ML model registry."""
    
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False)  # classification, regression, clustering, etc.
    algorithm = Column(String(100), nullable=False)  # xgboost, random_forest, etc.
    
    # Model configuration
    hyperparameters = Column(JSON, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    target_column = Column(String(100), nullable=True)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_score = Column(Float, nullable=True)
    
    # Model status
    is_active = Column(Boolean, default=True)
    is_training = Column(Boolean, default=False)
    last_trained = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional model data
    model_data = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<MLModel(id={self.id}, name='{self.name}', type='{self.model_type}')>"


class ModelVersion(Base):
    """Model version tracking."""
    
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)  # Reference to MLModel
    version = Column(String(20), nullable=False)
    
    # Version details
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    checksum = Column(String(64), nullable=True)
    
    # Training data
    training_data_path = Column(String(500), nullable=True)
    training_samples = Column(Integer, nullable=True)
    validation_samples = Column(Integer, nullable=True)
    
    # Performance metrics
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    
    # Model metadata
    training_duration_seconds = Column(Integer, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ModelVersion(id={self.id}, model_id={self.model_id}, version='{self.version}')>"


class PredictionLog(Base):
    """ML model prediction logging."""
    
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    model_version = Column(String(20), nullable=False)
    
    # Prediction details
    input_features = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=True)
    prediction_time_ms = Column(Float, nullable=True)
    
    # Context
    user_id = Column(Integer, nullable=True)
    session_id = Column(Integer, nullable=True)
    request_id = Column(String(100), nullable=True)
    
    # Ground truth (if available)
    actual_value = Column(JSON, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, model_id={self.model_id}, prediction_time={self.prediction_time_ms}ms)>"
