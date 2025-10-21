"""Question models for the adaptive learning assessment system."""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class QuestionBank(Base):
    """Question bank model for organizing questions."""
    
    __tablename__ = "question_banks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(100), nullable=False)
    grade_level = Column(String(20), nullable=False)
    
    # Bank configuration
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    questions = relationship("Question", back_populates="question_bank")
    creator = relationship("User", foreign_keys=[created_by])
    
    def __repr__(self):
        return f"<QuestionBank(id={self.id}, name='{self.name}', subject='{self.subject}')>"


class Question(Base):
    """Question model with IRT parameters."""
    
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    question_bank_id = Column(Integer, ForeignKey("question_banks.id"), nullable=False)
    
    # Question content
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)  # multiple_choice, true_false, fill_blank, etc.
    options = Column(JSON, nullable=True)  # For multiple choice questions
    correct_answer = Column(String(500), nullable=False)
    explanation = Column(Text, nullable=True)
    
    # Subject and difficulty
    subject = Column(String(100), nullable=False)
    topic = Column(String(100), nullable=True)
    subtopic = Column(String(100), nullable=True)
    grade_level = Column(String(20), nullable=False)
    
    # IRT parameters (calibrated through ML)
    difficulty = Column(Float, default=0.0)  # b parameter
    discrimination = Column(Float, default=1.0)  # a parameter
    guessing = Column(Float, default=0.0)  # c parameter
    
    # Additional parameters
    time_limit_seconds = Column(Integer, default=120)
    points = Column(Float, default=1.0)
    
    # Prerequisites and skills
    prerequisite_skills = Column(JSON, nullable=True)
    target_skills = Column(JSON, nullable=True)
    learning_objectives = Column(JSON, nullable=True)
    
    # Quality metrics
    usage_count = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    average_time = Column(Float, default=0.0)
    difficulty_rating = Column(Float, default=0.0)  # Human-rated difficulty
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Additional question data
    question_metadata = Column(JSON, nullable=True)
    
    # Relationships
    question_bank = relationship("QuestionBank", back_populates="questions")
    creator = relationship("User", foreign_keys=[created_by])
    responses = relationship("QuestionResponse", back_populates="question")
    
    def __repr__(self):
        return f"<Question(id={self.id}, subject='{self.subject}', difficulty={self.difficulty})>"


class QuestionResponse(Base):
    """Individual question response model."""
    
    __tablename__ = "question_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("assessment_sessions.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    
    # Response data
    user_answer = Column(String(500), nullable=False)
    is_correct = Column(Boolean, nullable=False)
    confidence_level = Column(Float, nullable=True)  # User's confidence (1-5 scale)
    
    # Time tracking
    time_spent_seconds = Column(Integer, nullable=False)
    time_to_first_response = Column(Integer, nullable=True)
    
    # Interaction data
    hints_used = Column(Integer, default=0)
    attempts_count = Column(Integer, default=1)
    skipped = Column(Boolean, default=False)
    
    # ML analysis
    predicted_difficulty = Column(Float, nullable=True)
    predicted_correctness = Column(Float, nullable=True)
    knowledge_state_before = Column(JSON, nullable=True)
    knowledge_state_after = Column(JSON, nullable=True)
    
    # Response metadata
    response_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("AssessmentSession", back_populates="responses")
    question = relationship("Question", back_populates="responses")
    
    def __repr__(self):
        return f"<QuestionResponse(id={self.id}, session_id={self.session_id}, is_correct={self.is_correct})>"
