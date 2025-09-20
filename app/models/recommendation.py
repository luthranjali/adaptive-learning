"""Recommendation models for the adaptive learning assessment system."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Recommendation(Base):
    """ML-generated recommendations model."""
    
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("assessment_sessions.id"), nullable=True)
    
    # Recommendation type
    recommendation_type = Column(String(50), nullable=False)  # course_level, learning_path, skill_focus
    subject = Column(String(100), nullable=False)
    
    # Recommendation content
    recommended_level = Column(String(20), nullable=True)  # beginner, intermediate, advanced
    confidence_score = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    
    # ML model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    model_confidence = Column(Float, nullable=True)
    
    # Performance metrics
    predicted_performance = Column(Float, nullable=True)
    success_probability = Column(Float, nullable=True)
    
    # Additional recommendations
    skill_gaps = Column(JSON, nullable=True)  # Identified skill gaps
    strengths = Column(JSON, nullable=True)  # Identified strengths
    next_steps = Column(JSON, nullable=True)  # Recommended next steps
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_accepted = Column(Boolean, nullable=True)
    feedback_score = Column(Float, nullable=True)  # User feedback on recommendation
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional recommendation data
    recommendation_data = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User")
    session = relationship("AssessmentSession")
    
    def __repr__(self):
        return f"<Recommendation(id={self.id}, type='{self.recommendation_type}', level='{self.recommended_level}')>"


class LearningPath(Base):
    """Personalized learning path model."""
    
    __tablename__ = "learning_paths"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject = Column(String(100), nullable=False)
    
    # Path configuration
    path_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    target_level = Column(String(20), nullable=False)
    estimated_duration_weeks = Column(Integer, nullable=True)
    
    # Path structure
    modules = Column(JSON, nullable=False)  # Ordered list of learning modules
    prerequisites = Column(JSON, nullable=True)  # Required prerequisites
    learning_objectives = Column(JSON, nullable=True)  # Learning objectives
    
    # Progress tracking
    current_module = Column(Integer, default=0)
    completion_percentage = Column(Float, default=0.0)
    is_completed = Column(Boolean, default=False)
    
    # ML-generated path
    ml_generated = Column(Boolean, default=False)
    model_name = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional path data
    path_data = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<LearningPath(id={self.id}, user_id={self.user_id}, subject='{self.subject}')>"
