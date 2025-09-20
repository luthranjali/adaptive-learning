"""Assessment API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models.user import User
from app.models.assessment import Assessment, AssessmentSession
from app.models.question import Question, QuestionResponse
from app.schemas.assessment import (
    AssessmentCreate, AssessmentResponse, AssessmentSessionCreate,
    AssessmentSessionResponse, QuestionResponseCreate, QuestionResponseResponse,
    AssessmentResult
)
from app.api.auth import get_current_user
from app.services.assessment_service import AssessmentService
from app.services.ml_service import MLService

router = APIRouter(prefix="/assessments", tags=["assessments"])


@router.post("/", response_model=AssessmentResponse)
async def create_assessment(
    assessment_data: AssessmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new assessment."""
    if current_user.role not in ["instructor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only instructors and admins can create assessments"
        )
    
    assessment = Assessment(
        **assessment_data.dict(),
        created_by=current_user.id
    )
    
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    
    return AssessmentResponse.from_orm(assessment)


@router.get("/", response_model=List[AssessmentResponse])
async def get_assessments(
    subject: Optional[str] = None,
    grade_level: Optional[str] = None,
    difficulty_level: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get available assessments with optional filters."""
    query = db.query(Assessment).filter(Assessment.is_active == True)
    
    if subject:
        query = query.filter(Assessment.subject == subject)
    if grade_level:
        query = query.filter(Assessment.grade_level == grade_level)
    if difficulty_level:
        query = query.filter(Assessment.difficulty_level == difficulty_level)
    
    assessments = query.all()
    return [AssessmentResponse.from_orm(assessment) for assessment in assessments]


@router.get("/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(assessment_id: int, db: Session = Depends(get_db)):
    """Get a specific assessment."""
    assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    return AssessmentResponse.from_orm(assessment)


@router.post("/start", response_model=AssessmentSessionResponse)
async def start_assessment(
    session_data: AssessmentSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new assessment session."""
    # Check if assessment exists
    assessment = db.query(Assessment).filter(Assessment.id == session_data.assessment_id).first()
    if not assessment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found"
        )
    
    # Check if user has active session for this assessment
    active_session = db.query(AssessmentSession).filter(
        AssessmentSession.user_id == current_user.id,
        AssessmentSession.assessment_id == session_data.assessment_id,
        AssessmentSession.status == "in_progress"
    ).first()
    
    if active_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assessment already in progress"
        )
    
    # Create new session
    session = AssessmentSession(
        user_id=current_user.id,
        assessment_id=session_data.assessment_id,
        status="in_progress"
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return AssessmentSessionResponse.from_orm(session)


@router.get("/sessions/{session_id}", response_model=AssessmentSessionResponse)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get assessment session details."""
    session = db.query(AssessmentSession).filter(
        AssessmentSession.id == session_id,
        AssessmentSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return AssessmentSessionResponse.from_orm(session)


@router.get("/sessions/{session_id}/next-question")
async def get_next_question(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get next question for assessment session."""
    session = db.query(AssessmentSession).filter(
        AssessmentSession.id == session_id,
        AssessmentSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if session.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session not in progress"
        )
    
    # Use ML service to select next question
    ml_service = MLService()
    question = await ml_service.select_next_question(session, db)
    
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No more questions available"
        )
    
    return question


@router.post("/sessions/{session_id}/answer", response_model=QuestionResponseResponse)
async def submit_answer(
    session_id: int,
    response_data: QuestionResponseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit answer for a question."""
    session = db.query(AssessmentSession).filter(
        AssessmentSession.id == session_id,
        AssessmentSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if session.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session not in progress"
        )
    
    # Create response record
    response = QuestionResponse(
        session_id=session_id,
        **response_data.dict()
    )
    
    db.add(response)
    
    # Update session statistics
    session.total_questions += 1
    if response_data.is_correct:
        session.correct_answers += 1
    
    session.time_spent_seconds += response_data.time_spent_seconds
    session.average_time_per_question = session.time_spent_seconds / session.total_questions
    
    # Calculate scores
    session.total_score = session.correct_answers
    session.percentage_score = (session.correct_answers / session.total_questions) * 100
    
    db.commit()
    db.refresh(response)
    
    # Update ML models with response
    ml_service = MLService()
    await ml_service.update_with_response(session, response, db)
    
    return QuestionResponseResponse.from_orm(response)


@router.post("/sessions/{session_id}/complete", response_model=AssessmentResult)
async def complete_assessment(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Complete assessment session and get results."""
    session = db.query(AssessmentSession).filter(
        AssessmentSession.id == session_id,
        AssessmentSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if session.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session not in progress"
        )
    
    # Update session status
    session.status = "completed"
    session.completed_at = datetime.utcnow()
    
    # Generate ML analysis
    ml_service = MLService()
    analysis_results = await ml_service.analyze_session(session, db)
    
    # Update session with analysis results
    session.knowledge_state = analysis_results.get('knowledge_state')
    session.difficulty_progression = analysis_results.get('difficulty_progression')
    session.learning_patterns = analysis_results.get('learning_patterns')
    session.skill_mastery = analysis_results.get('skill_mastery')
    
    db.commit()
    
    # Generate recommendations
    recommendations = await ml_service.generate_recommendations(session, db)
    
    return AssessmentResult(
        session_id=session.id,
        total_questions=session.total_questions,
        correct_answers=session.correct_answers,
        percentage_score=session.percentage_score,
        time_spent_seconds=session.time_spent_seconds,
        knowledge_state=session.knowledge_state,
        difficulty_progression=session.difficulty_progression,
        learning_patterns=session.learning_patterns,
        skill_mastery=session.skill_mastery,
        recommendations=recommendations
    )


@router.get("/sessions/", response_model=List[AssessmentSessionResponse])
async def get_user_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's assessment sessions."""
    sessions = db.query(AssessmentSession).filter(
        AssessmentSession.user_id == current_user.id
    ).order_by(AssessmentSession.started_at.desc()).all()
    
    return [AssessmentSessionResponse.from_orm(session) for session in sessions]


@router.get("/sessions/{session_id}/results", response_model=AssessmentResult)
async def get_session_results(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed results for a completed session."""
    session = db.query(AssessmentSession).filter(
        AssessmentSession.id == session_id,
        AssessmentSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session not completed"
        )
    
    return AssessmentResult(
        session_id=session.id,
        total_questions=session.total_questions,
        correct_answers=session.correct_answers,
        percentage_score=session.percentage_score,
        time_spent_seconds=session.time_spent_seconds,
        knowledge_state=session.knowledge_state,
        difficulty_progression=session.difficulty_progression,
        learning_patterns=session.learning_patterns,
        skill_mastery=session.skill_mastery,
        recommendations=session.session_data.get('recommendations') if session.session_data else None
    )
