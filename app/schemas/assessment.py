"""Assessment schemas for API serialization."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class AssessmentBase(BaseModel):
    """Base assessment schema."""
    name: str
    description: Optional[str] = None
    subject: str
    grade_level: str
    difficulty_level: str
    max_questions: int = 50
    time_limit_minutes: int = 60
    passing_score: float = 70.0


class AssessmentCreate(AssessmentBase):
    """Schema for assessment creation."""
    pass


class AssessmentResponse(AssessmentBase):
    """Schema for assessment response."""
    id: int
    is_active: bool
    created_by: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AssessmentSessionBase(BaseModel):
    """Base assessment session schema."""
    assessment_id: int


class AssessmentSessionCreate(AssessmentSessionBase):
    """Schema for assessment session creation."""
    pass


class AssessmentSessionResponse(AssessmentSessionBase):
    """Schema for assessment session response."""
    id: int
    user_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_questions: int
    correct_answers: int
    total_score: float
    percentage_score: float
    time_spent_seconds: int
    average_time_per_question: float
    
    class Config:
        from_attributes = True


class QuestionResponseBase(BaseModel):
    """Base question response schema."""
    question_id: int
    user_answer: str
    is_correct: bool
    confidence_level: Optional[float] = None
    time_spent_seconds: int
    hints_used: int = 0
    attempts_count: int = 1
    skipped: bool = False


class QuestionResponseCreate(QuestionResponseBase):
    """Schema for question response creation."""
    pass


class QuestionResponseResponse(QuestionResponseBase):
    """Schema for question response response."""
    id: int
    session_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class AssessmentResult(BaseModel):
    """Schema for assessment results."""
    session_id: int
    total_questions: int
    correct_answers: int
    percentage_score: float
    time_spent_seconds: int
    knowledge_state: Optional[Dict[str, Any]] = None
    difficulty_progression: Optional[List[float]] = None
    learning_patterns: Optional[Dict[str, Any]] = None
    skill_mastery: Optional[Dict[str, float]] = None
    recommendations: Optional[Dict[str, Any]] = None
