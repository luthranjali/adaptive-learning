"""Assessment service for business logic."""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.models.assessment import Assessment, AssessmentSession
from app.models.question import Question, QuestionResponse
from app.models.user import User


class AssessmentService:
    """Service for assessment-related business logic."""
    
    def __init__(self):
        """Initialize assessment service."""
        pass
    
    def create_assessment(self, assessment_data: Dict, creator_id: int, db: Session) -> Assessment:
        """Create a new assessment."""
        assessment = Assessment(
            name=assessment_data['name'],
            description=assessment_data.get('description'),
            subject=assessment_data['subject'],
            grade_level=assessment_data['grade_level'],
            difficulty_level=assessment_data['difficulty_level'],
            max_questions=assessment_data.get('max_questions', 50),
            time_limit_minutes=assessment_data.get('time_limit_minutes', 60),
            passing_score=assessment_data.get('passing_score', 70.0),
            created_by=creator_id
        )
        
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        
        return assessment
    
    def start_assessment_session(self, user_id: int, assessment_id: int, db: Session) -> AssessmentSession:
        """Start a new assessment session."""
        # Check if user has active session
        active_session = db.query(AssessmentSession).filter(
            AssessmentSession.user_id == user_id,
            AssessmentSession.assessment_id == assessment_id,
            AssessmentSession.status == "in_progress"
        ).first()
        
        if active_session:
            raise ValueError("Assessment already in progress")
        
        # Create new session
        session = AssessmentSession(
            user_id=user_id,
            assessment_id=assessment_id,
            status="in_progress"
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session
    
    def submit_answer(self, session_id: int, question_id: int, 
                     user_answer: str, time_spent: int, 
                     confidence_level: Optional[float] = None,
                     hints_used: int = 0, db: Session = None) -> QuestionResponse:
        """Submit an answer for a question."""
        # Get session
        session = db.query(AssessmentSession).filter(AssessmentSession.id == session_id).first()
        if not session:
            raise ValueError("Session not found")
        
        if session.status != "in_progress":
            raise ValueError("Session not in progress")
        
        # Get question to check correct answer
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question:
            raise ValueError("Question not found")
        
        # Determine if answer is correct
        is_correct = self._check_answer(question, user_answer)
        
        # Create response
        response = QuestionResponse(
            session_id=session_id,
            question_id=question_id,
            user_answer=user_answer,
            is_correct=is_correct,
            confidence_level=confidence_level,
            time_spent_seconds=time_spent,
            hints_used=hints_used
        )
        
        db.add(response)
        
        # Update session statistics
        session.total_questions += 1
        if is_correct:
            session.correct_answers += 1
        
        session.time_spent_seconds += time_spent
        session.average_time_per_question = session.time_spent_seconds / session.total_questions
        
        # Calculate scores
        session.total_score = session.correct_answers
        session.percentage_score = (session.correct_answers / session.total_questions) * 100
        
        db.commit()
        db.refresh(response)
        
        return response
    
    def _check_answer(self, question: Question, user_answer: str) -> bool:
        """Check if user answer is correct."""
        if question.question_type == "multiple_choice":
            return user_answer.strip().lower() == question.correct_answer.strip().lower()
        elif question.question_type == "true_false":
            return user_answer.strip().lower() == question.correct_answer.strip().lower()
        elif question.question_type == "fill_blank":
            # For fill in the blank, check if answer contains key terms
            correct_terms = question.correct_answer.lower().split()
            user_terms = user_answer.lower().split()
            return any(term in user_terms for term in correct_terms)
        else:
            # Default exact match
            return user_answer.strip().lower() == question.correct_answer.strip().lower()
    
    def complete_assessment_session(self, session_id: int, db: Session) -> AssessmentSession:
        """Complete an assessment session."""
        session = db.query(AssessmentSession).filter(AssessmentSession.id == session_id).first()
        if not session:
            raise ValueError("Session not found")
        
        if session.status != "in_progress":
            raise ValueError("Session not in progress")
        
        # Update session status
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(session)
        
        return session
    
    def get_assessment_results(self, session_id: int, db: Session) -> Dict[str, Any]:
        """Get detailed results for an assessment session."""
        session = db.query(AssessmentSession).filter(AssessmentSession.id == session_id).first()
        if not session:
            raise ValueError("Session not found")
        
        if session.status != "completed":
            raise ValueError("Session not completed")
        
        # Get all responses
        responses = db.query(QuestionResponse).filter(
            QuestionResponse.session_id == session_id
        ).all()
        
        # Calculate detailed statistics
        results = {
            'session_id': session.id,
            'assessment_name': session.assessment.name,
            'total_questions': session.total_questions,
            'correct_answers': session.correct_answers,
            'incorrect_answers': session.total_questions - session.correct_answers,
            'percentage_score': session.percentage_score,
            'time_spent_seconds': session.time_spent_seconds,
            'average_time_per_question': session.average_time_per_question,
            'started_at': session.started_at,
            'completed_at': session.completed_at,
            'duration_minutes': (session.completed_at - session.started_at).total_seconds() / 60,
            'responses': []
        }
        
        # Add response details
        for response in responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            results['responses'].append({
                'question_id': response.question_id,
                'question_text': question.question_text if question else "Question not found",
                'user_answer': response.user_answer,
                'correct_answer': question.correct_answer if question else "N/A",
                'is_correct': response.is_correct,
                'time_spent_seconds': response.time_spent_seconds,
                'confidence_level': response.confidence_level,
                'hints_used': response.hints_used
            })
        
        return results
    
    def get_user_assessment_history(self, user_id: int, db: Session, 
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's assessment history."""
        sessions = db.query(AssessmentSession).filter(
            AssessmentSession.user_id == user_id,
            AssessmentSession.status == "completed"
        ).order_by(AssessmentSession.completed_at.desc()).limit(limit).all()
        
        history = []
        for session in sessions:
            history.append({
                'session_id': session.id,
                'assessment_name': session.assessment.name,
                'subject': session.assessment.subject,
                'grade_level': session.assessment.grade_level,
                'percentage_score': session.percentage_score,
                'total_questions': session.total_questions,
                'time_spent_minutes': session.time_spent_seconds / 60,
                'completed_at': session.completed_at,
                'knowledge_state': session.knowledge_state,
                'skill_mastery': session.skill_mastery
            })
        
        return history
    
    def get_assessment_statistics(self, assessment_id: int, db: Session) -> Dict[str, Any]:
        """Get statistics for an assessment."""
        # Get all completed sessions for this assessment
        sessions = db.query(AssessmentSession).filter(
            AssessmentSession.assessment_id == assessment_id,
            AssessmentSession.status == "completed"
        ).all()
        
        if not sessions:
            return {'message': 'No completed sessions found'}
        
        # Calculate statistics
        scores = [session.percentage_score for session in sessions]
        times = [session.time_spent_seconds / 60 for session in sessions]  # Convert to minutes
        
        statistics = {
            'total_sessions': len(sessions),
            'average_score': sum(scores) / len(scores),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'average_time_minutes': sum(times) / len(times),
            'fastest_time_minutes': min(times),
            'slowest_time_minutes': max(times),
            'completion_rate': len(sessions) / len(sessions) * 100,  # All sessions are completed
            'score_distribution': {
                '90-100': len([s for s in scores if s >= 90]),
                '80-89': len([s for s in scores if 80 <= s < 90]),
                '70-79': len([s for s in scores if 70 <= s < 80]),
                '60-69': len([s for s in scores if 60 <= s < 69]),
                'below_60': len([s for s in scores if s < 60])
            }
        }
        
        return statistics
    
    def get_question_bank_statistics(self, question_bank_id: int, db: Session) -> Dict[str, Any]:
        """Get statistics for a question bank."""
        # Get all questions in the bank
        questions = db.query(Question).filter(
            Question.question_bank_id == question_bank_id
        ).all()
        
        if not questions:
            return {'message': 'No questions found in bank'}
        
        # Get all responses for these questions
        question_ids = [q.id for q in questions]
        responses = db.query(QuestionResponse).filter(
            QuestionResponse.question_id.in_(question_ids)
        ).all()
        
        # Calculate statistics
        question_stats = {}
        for question in questions:
            question_responses = [r for r in responses if r.question_id == question.id]
            
            if question_responses:
                correct_count = sum(1 for r in question_responses if r.is_correct)
                total_count = len(question_responses)
                avg_time = sum(r.time_spent_seconds for r in question_responses) / total_count
                
                question_stats[question.id] = {
                    'question_text': question.question_text[:100] + "...",  # Truncate
                    'difficulty': question.difficulty,
                    'total_attempts': total_count,
                    'correct_attempts': correct_count,
                    'success_rate': correct_count / total_count,
                    'average_time_seconds': avg_time
                }
        
        return {
            'total_questions': len(questions),
            'total_responses': len(responses),
            'question_statistics': question_stats
        }
