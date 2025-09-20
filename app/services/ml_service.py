"""ML service for handling machine learning operations."""

import numpy as np
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from datetime import datetime
import asyncio

from app.models.assessment import AssessmentSession
from app.models.question import Question, QuestionResponse
from app.ml.models import (
    IRTModel, BKTModel, LevelRecommender, 
    QuestionSelector, GapDetector, PerformanceAnalyzer
)
from app.config import settings


class MLService:
    """Service for managing ML models and operations."""
    
    def __init__(self):
        """Initialize ML service with models."""
        self.irt_model = IRTModel()
        self.bkt_model = BKTModel()
        self.level_recommender = LevelRecommender()
        self.question_selector = QuestionSelector()
        self.gap_detector = GapDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from storage."""
        try:
            # Load models from files
            self.irt_model.load_model(f"{settings.model_storage_path}/irt_model.pkl")
            self.bkt_model.load_model(f"{settings.model_storage_path}/bkt_model.pkl")
            self.level_recommender.load_model(f"{settings.model_storage_path}/level_recommender.pkl")
            self.question_selector.load_model(f"{settings.model_storage_path}/question_selector.pkl")
            self.gap_detector.load_model(f"{settings.model_storage_path}/gap_detector.pkl")
            self.performance_analyzer.load_model(f"{settings.model_storage_path}/performance_analyzer.pkl")
        except FileNotFoundError:
            # Models not found, will use default parameters
            pass
    
    def _save_models(self):
        """Save trained models to storage."""
        import os
        os.makedirs(settings.model_storage_path, exist_ok=True)
        
        self.irt_model.save_model(f"{settings.model_storage_path}/irt_model.pkl")
        self.bkt_model.save_model(f"{settings.model_storage_path}/bkt_model.pkl")
        self.level_recommender.save_model(f"{settings.model_storage_path}/level_recommender.pkl")
        self.question_selector.save_model(f"{settings.model_storage_path}/question_selector.pkl")
        self.gap_detector.save_model(f"{settings.model_storage_path}/gap_detector.pkl")
        self.performance_analyzer.save_model(f"{settings.model_storage_path}/performance_analyzer.pkl")
    
    async def select_next_question(self, session: AssessmentSession, db: Session) -> Optional[Dict]:
        """Select next question using ML models."""
        # Get available questions for the assessment
        assessment = session.assessment
        available_questions = db.query(Question).filter(
            Question.question_bank_id.in_(
                db.query(QuestionBank.id).filter(QuestionBank.subject == assessment.subject)
            ),
            Question.grade_level == assessment.grade_level,
            Question.is_active == True
        ).all()
        
        if not available_questions:
            return None
        
        # Convert to question dictionaries
        question_data = []
        for q in available_questions:
            question_data.append({
                'id': q.id,
                'difficulty': q.difficulty,
                'subject': q.subject,
                'topic': q.topic,
                'question_text': q.question_text,
                'question_type': q.question_type,
                'options': q.options
            })
        
        # Get student ability estimate
        student_ability = await self._estimate_student_ability(session, db)
        
        # Create assessment context
        assessment_context = {
            'target_subjects': [assessment.subject],
            'target_topics': [],  # Could be extracted from learning objectives
            'recent_questions': [r.question_id for r in session.responses[-5:]]  # Last 5 questions
        }
        
        # Select next question
        selected_question = self.question_selector.select_next_question(
            question_data, student_ability, assessment_context
        )
        
        return selected_question
    
    async def _estimate_student_ability(self, session: AssessmentSession, db: Session) -> float:
        """Estimate student ability using IRT model."""
        if not session.responses:
            return 0.0  # Default ability for new students
        
        # Get responses and question difficulties
        responses = {}
        question_ids = []
        
        for response in session.responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            if question:
                responses[question.id] = response.is_correct
                question_ids.append(question.id)
        
        if not responses:
            return 0.0
        
        # Estimate ability using IRT model
        ability = self.irt_model.estimate_ability(responses, question_ids)
        return ability
    
    async def update_with_response(self, session: AssessmentSession, response: QuestionResponse, db: Session):
        """Update ML models with new response."""
        # Update question selector
        self.question_selector.update_with_response(
            response.question_id,
            response.is_correct,
            response.time_spent_seconds
        )
        
        # Update IRT model (if enough data)
        if len(session.responses) >= 5:
            await self._update_irt_model(session, db)
        
        # Update BKT model
        await self._update_bkt_model(session, response, db)
    
    async def _update_irt_model(self, session: AssessmentSession, db: Session):
        """Update IRT model with session data."""
        # Collect response data
        responses = []
        thetas = []
        question_ids = []
        
        for response in session.responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            if question:
                responses.append(response.is_correct)
                thetas.append(0.0)  # Could use previous ability estimate
                question_ids.append(question.id)
        
        if len(responses) >= 5:
            # Fit IRT model
            responses_array = np.array(responses).reshape(1, -1)
            thetas_array = np.array(thetas)
            
            self.irt_model.fit(responses_array, thetas_array, question_ids)
    
    async def _update_bkt_model(self, session: AssessmentSession, response: QuestionResponse, db: Session):
        """Update BKT model with response data."""
        question = db.query(Question).filter(Question.id == response.question_id).first()
        if not question or not question.target_skills:
            return
        
        # Update skill mastery for each target skill
        for skill in question.target_skills:
            # Get existing skill data for this student
            skill_data = self._get_skill_data(session.user_id, skill, db)
            
            # Add new response
            skill_data.append((response.is_correct, response.time_spent_seconds))
            
            # Update BKT model
            self.bkt_model.fit({skill: skill_data})
    
    def _get_skill_data(self, user_id: int, skill: str, db: Session) -> List[tuple]:
        """Get historical skill data for a user."""
        # This would query the database for historical responses
        # For now, return empty list
        return []
    
    async def analyze_session(self, session: AssessmentSession, db: Session) -> Dict[str, Any]:
        """Analyze completed session using ML models."""
        analysis_results = {}
        
        # Knowledge state analysis using BKT
        knowledge_state = await self._analyze_knowledge_state(session, db)
        analysis_results['knowledge_state'] = knowledge_state
        
        # Difficulty progression analysis
        difficulty_progression = await self._analyze_difficulty_progression(session, db)
        analysis_results['difficulty_progression'] = difficulty_progression
        
        # Learning patterns analysis
        learning_patterns = await self._analyze_learning_patterns(session, db)
        analysis_results['learning_patterns'] = learning_patterns
        
        # Skill mastery analysis
        skill_mastery = await self._analyze_skill_mastery(session, db)
        analysis_results['skill_mastery'] = skill_mastery
        
        return analysis_results
    
    async def _analyze_knowledge_state(self, session: AssessmentSession, db: Session) -> Dict[str, Any]:
        """Analyze knowledge state using BKT model."""
        # Get skill-specific responses
        skill_responses = {}
        
        for response in session.responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            if question and question.target_skills:
                for skill in question.target_skills:
                    if skill not in skill_responses:
                        skill_responses[skill] = []
                    skill_responses[skill].append(response.is_correct)
        
        # Analyze each skill
        skill_analysis = {}
        for skill, responses in skill_responses.items():
            mastery_info = self.bkt_model.get_skill_mastery(skill, responses)
            skill_analysis[skill] = mastery_info
        
        # Overall knowledge state
        overall_mastery = np.mean([info['current_mastery'] for info in skill_analysis.values()]) if skill_analysis else 0.0
        
        return {
            'overall_mastery': overall_mastery,
            'skill_count': len(skill_analysis),
            'mastered_skills': len([info for info in skill_analysis.values() if info['current_mastery'] > 0.8]),
            'learning_velocity': np.mean([info['learning_gain'] for info in skill_analysis.values()]) if skill_analysis else 0.0,
            'skill_details': skill_analysis
        }
    
    async def _analyze_difficulty_progression(self, session: AssessmentSession, db: Session) -> List[float]:
        """Analyze difficulty progression throughout session."""
        progression = []
        
        for response in session.responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            if question:
                progression.append(question.difficulty)
        
        return progression
    
    async def _analyze_learning_patterns(self, session: AssessmentSession, db: Session) -> Dict[str, Any]:
        """Analyze learning patterns from session data."""
        patterns = {}
        
        # Consistency analysis
        correct_responses = [r.is_correct for r in session.responses]
        patterns['consistency_score'] = np.mean(correct_responses) if correct_responses else 0.0
        
        # Improvement over time
        if len(correct_responses) > 1:
            first_half = correct_responses[:len(correct_responses)//2]
            second_half = correct_responses[len(correct_responses)//2:]
            patterns['improvement_rate'] = np.mean(second_half) - np.mean(first_half)
        else:
            patterns['improvement_rate'] = 0.0
        
        # Time patterns
        time_spent = [r.time_spent_seconds for r in session.responses]
        patterns['time_consistency'] = 1.0 - (np.std(time_spent) / np.mean(time_spent)) if time_spent and np.mean(time_spent) > 0 else 0.0
        
        # Engagement score
        patterns['engagement_score'] = min(1.0, len(session.responses) / session.assessment.max_questions)
        
        return patterns
    
    async def _analyze_skill_mastery(self, session: AssessmentSession, db: Session) -> Dict[str, float]:
        """Analyze skill mastery levels."""
        skill_mastery = {}
        
        # Group responses by skill
        skill_responses = {}
        for response in session.responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            if question and question.target_skills:
                for skill in question.target_skills:
                    if skill not in skill_responses:
                        skill_responses[skill] = []
                    skill_responses[skill].append(response.is_correct)
        
        # Calculate mastery for each skill
        for skill, responses in skill_responses.items():
            mastery_info = self.bkt_model.get_skill_mastery(skill, responses)
            skill_mastery[skill] = mastery_info['current_mastery']
        
        return skill_mastery
    
    async def generate_recommendations(self, session: AssessmentSession, db: Session) -> Dict[str, Any]:
        """Generate recommendations based on session analysis."""
        recommendations = {}
        
        # Prepare assessment data for level recommendation
        assessment_data = {
            'total_score': session.total_score,
            'percentage_score': session.percentage_score,
            'correct_answers': session.correct_answers,
            'total_questions': session.total_questions,
            'time_spent_seconds': session.time_spent_seconds,
            'average_time_per_question': session.average_time_per_question,
            'knowledge_state': session.knowledge_state,
            'learning_patterns': session.learning_patterns,
            'skill_mastery': session.skill_mastery
        }
        
        # Generate level recommendation
        level_recommendation = self.level_recommender.predict_level(assessment_data)
        recommendations['level_recommendation'] = level_recommendation
        
        # Generate skill-specific recommendations
        skill_recommendations = []
        if session.skill_mastery:
            for skill, mastery in session.skill_mastery.items():
                if mastery < 0.6:  # Skills needing improvement
                    skill_recommendations.append({
                        'skill': skill,
                        'mastery_level': mastery,
                        'recommendation': f"Focus on {skill} - current mastery: {mastery:.2f}"
                    })
        
        recommendations['skill_recommendations'] = skill_recommendations
        
        # Generate learning path recommendations
        recommendations['learning_path'] = {
            'next_focus_areas': [rec['skill'] for rec in skill_recommendations[:3]],
            'suggested_difficulty': 'intermediate' if session.percentage_score > 70 else 'beginner',
            'study_recommendations': [
                "Practice with similar difficulty questions",
                "Review foundational concepts",
                "Focus on time management"
            ]
        }
        
        return recommendations
    
    async def train_models(self, training_data: List[Dict], db: Session):
        """Train ML models with new data."""
        # This would implement model training with collected data
        # For now, just save the current models
        self._save_models()
    
    async def get_model_insights(self) -> Dict[str, Any]:
        """Get insights about ML model performance."""
        insights = {}
        
        # Question selector insights
        insights['question_selector'] = self.question_selector.get_learning_insights(0.0)
        
        # Gap detector insights
        insights['gap_detector'] = self.gap_detector.get_cluster_insights()
        
        # Level recommender insights
        insights['level_recommender'] = {
            'feature_importance': self.level_recommender.get_feature_importance()
        }
        
        return insights
