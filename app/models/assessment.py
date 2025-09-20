"""Assessment models for the adaptive learning assessment system."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Assessment(Base):
    """Assessment template model."""
    
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(100), nullable=False)
    grade_level = Column(String(20), nullable=False)
    difficulty_level = Column(String(20), nullable=False)  # beginner, intermediate, advanced
    
    # Assessment configuration
    max_questions = Column(Integer, default=50)
    time_limit_minutes = Column(Integer, default=60)
    passing_score = Column(Float, default=70.0)
    
    # IRT parameters
    irt_difficulty_mean = Column(Float, default=0.0)
    irt_difficulty_std = Column(Float, default=1.0)
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional configuration
    config = Column(JSON, nullable=True)
    
    # Relationships
    sessions = relationship("AssessmentSession", back_populates="assessment")
    creator = relationship("User", foreign_keys=[created_by])
    
    def __repr__(self):
        return f"<Assessment(id={self.id}, name='{self.name}', subject='{self.subject}')>"


class AssessmentSession(Base):
    """Individual assessment session model."""
    
    __tablename__ = "assessment_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assessment_id = Column(Integer, ForeignKey("assessments.id"), nullable=False)
    
    # Session status
    status = Column(String(20), default="in_progress")  # in_progress, completed, abandoned
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Performance metrics
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    total_score = Column(Float, default=0.0)
    percentage_score = Column(Float, default=0.0)
    
    # Time tracking
    time_spent_seconds = Column(Integer, default=0)
    average_time_per_question = Column(Float, default=0.0)
    
    # ML analysis results
    knowledge_state = Column(JSON, nullable=True)  # BKT results
    difficulty_progression = Column(JSON, nullable=True)  # IRT analysis
    learning_patterns = Column(JSON, nullable=True)  # Pattern recognition results
    skill_mastery = Column(JSON, nullable=True)  # Skill-level mastery scores
    
    # Adaptive parameters
    current_difficulty = Column(Float, default=0.0)
    confidence_level = Column(Float, default=0.0)
    next_question_hint = Column(JSON, nullable=True)  # ML recommendation for next question
    
    # Additional session data
    session_data = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User")
    assessment = relationship("Assessment", back_populates="sessions")
    responses = relationship("QuestionResponse", back_populates="session")
    
    def __repr__(self):
        return f"<AssessmentSession(id={self.id}, user_id={self.user_id}, status='{self.status}')>"
