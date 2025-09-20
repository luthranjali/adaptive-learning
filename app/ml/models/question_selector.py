"""Adaptive question selection using Thompson Sampling."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from scipy.stats import beta
import random


class QuestionSelector:
    """Thompson Sampling-based adaptive question selector."""
    
    def __init__(self, exploration_factor: float = 0.1):
        """Initialize question selector with Thompson Sampling."""
        self.exploration_factor = exploration_factor
        self.question_stats = {}  # question_id -> {alpha, beta, difficulty, subject}
        self.student_ability = 0.0
        self.is_initialized = False
        
    def _initialize_question(self, question_id: int, difficulty: float, 
                           subject: str, topic: str = None) -> None:
        """Initialize question statistics."""
        self.question_stats[question_id] = {
            'alpha': 1.0,  # Success count + 1
            'beta': 1.0,   # Failure count + 1
            'difficulty': difficulty,
            'subject': subject,
            'topic': topic,
            'total_attempts': 0,
            'success_rate': 0.5
        }
    
    def _update_question_stats(self, question_id: int, response: bool, 
                             time_spent: float) -> None:
        """Update question statistics based on response."""
        if question_id not in self.question_stats:
            return
        
        stats = self.question_stats[question_id]
        stats['total_attempts'] += 1
        
        if response:
            stats['alpha'] += 1
        else:
            stats['beta'] += 1
        
        # Update success rate
        stats['success_rate'] = stats['alpha'] / (stats['alpha'] + stats['beta'])
    
    def _thompson_sampling_score(self, question_id: int, student_ability: float) -> float:
        """Calculate Thompson Sampling score for question selection."""
        if question_id not in self.question_stats:
            return 0.5  # Default score for unknown questions
        
        stats = self.question_stats[question_id]
        
        # Sample from Beta distribution
        sampled_success_rate = beta.rvs(stats['alpha'], stats['beta'])
        
        # Adjust for difficulty vs ability match
        difficulty = stats['difficulty']
        ability_match = 1.0 - abs(student_ability - difficulty) / 6.0  # Normalize to [0,1]
        ability_match = max(0.0, min(1.0, ability_match))
        
        # Combine Thompson sampling with ability matching
        score = (1 - self.exploration_factor) * sampled_success_rate + \
                self.exploration_factor * ability_match
        
        return score
    
    def _calculate_question_utility(self, question_id: int, student_ability: float,
                                  assessment_context: Dict) -> float:
        """Calculate utility score for question selection."""
        if question_id not in self.question_stats:
            return 0.5
        
        stats = self.question_stats[question_id]
        
        # Base Thompson sampling score
        thompson_score = self._thompson_sampling_score(question_id, student_ability)
        
        # Difficulty appropriateness
        difficulty = stats['difficulty']
        difficulty_penalty = abs(student_ability - difficulty) / 3.0
        difficulty_score = max(0.0, 1.0 - difficulty_penalty)
        
        # Subject coverage (if specified in context)
        target_subjects = assessment_context.get('target_subjects', [])
        subject_bonus = 1.0
        if target_subjects and stats['subject'] in target_subjects:
            subject_bonus = 1.2
        
        # Topic coverage (if specified in context)
        target_topics = assessment_context.get('target_topics', [])
        topic_bonus = 1.0
        if target_topics and stats.get('topic') in target_topics:
            topic_bonus = 1.1
        
        # Avoid recently used questions
        recent_questions = assessment_context.get('recent_questions', [])
        recency_penalty = 1.0
        if question_id in recent_questions:
            recency_penalty = 0.5
        
        # Balance exploration vs exploitation
        exploration_bonus = 1.0
        if stats['total_attempts'] < 5:  # Encourage exploration of new questions
            exploration_bonus = 1.3
        
        # Calculate final utility
        utility = (thompson_score * 0.4 + 
                  difficulty_score * 0.3 + 
                  (stats['success_rate'] * 0.3)) * \
                  subject_bonus * topic_bonus * recency_penalty * exploration_bonus
        
        return utility
    
    def select_next_question(self, available_questions: List[Dict], 
                           student_ability: float,
                           assessment_context: Dict) -> Optional[Dict]:
        """
        Select next question using Thompson Sampling.
        
        Args:
            available_questions: List of available question dictionaries
            student_ability: Current estimate of student ability
            assessment_context: Context information for assessment
            
        Returns:
            Selected question dictionary or None
        """
        if not available_questions:
            return None
        
        self.student_ability = student_ability
        
        # Initialize questions if not already done
        for q in available_questions:
            if q['id'] not in self.question_stats:
                self._initialize_question(
                    q['id'], 
                    q.get('difficulty', 0.0),
                    q.get('subject', 'unknown'),
                    q.get('topic')
                )
        
        # Calculate utility scores for all available questions
        question_utilities = []
        for question in available_questions:
            utility = self._calculate_question_utility(
                question['id'], student_ability, assessment_context
            )
            question_utilities.append((question, utility))
        
        # Sort by utility (descending)
        question_utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Select top question
        selected_question, utility_score = question_utilities[0]
        
        # Add selection metadata
        selected_question['selection_utility'] = utility_score
        selected_question['thompson_score'] = self._thompson_sampling_score(
            selected_question['id'], student_ability
        )
        
        return selected_question
    
    def update_with_response(self, question_id: int, response: bool, 
                           time_spent: float, student_ability: float = None) -> None:
        """Update model with student response."""
        self._update_question_stats(question_id, response, time_spent)
        
        if student_ability is not None:
            self.student_ability = student_ability
    
    def get_question_recommendations(self, available_questions: List[Dict],
                                   student_ability: float,
                                   assessment_context: Dict,
                                   n_recommendations: int = 5) -> List[Dict]:
        """Get top N question recommendations."""
        if not available_questions:
            return []
        
        # Calculate utilities for all questions
        question_utilities = []
        for question in available_questions:
            utility = self._calculate_question_utility(
                question['id'], student_ability, assessment_context
            )
            question_utilities.append((question, utility))
        
        # Sort and return top N
        question_utilities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for question, utility in question_utilities[:n_recommendations]:
            question_copy = question.copy()
            question_copy['recommendation_score'] = utility
            question_copy['thompson_score'] = self._thompson_sampling_score(
                question['id'], student_ability
            )
            recommendations.append(question_copy)
        
        return recommendations
    
    def get_question_statistics(self, question_id: int) -> Optional[Dict]:
        """Get statistics for a specific question."""
        if question_id not in self.question_stats:
            return None
        
        stats = self.question_stats[question_id]
        return {
            'question_id': question_id,
            'total_attempts': stats['total_attempts'],
            'success_rate': stats['success_rate'],
            'alpha': stats['alpha'],
            'beta': stats['beta'],
            'difficulty': stats['difficulty'],
            'subject': stats['subject'],
            'topic': stats.get('topic'),
            'confidence_interval': self._get_confidence_interval(stats['alpha'], stats['beta'])
        }
    
    def _get_confidence_interval(self, alpha: float, beta: float, 
                               confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for success rate."""
        # Use Beta distribution percentiles
        lower = beta.ppf((1 - confidence) / 2, alpha, beta)
        upper = beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
        return (lower, upper)
    
    def get_learning_insights(self, student_ability: float) -> Dict[str, Any]:
        """Get insights about student learning patterns."""
        if not self.question_stats:
            return {'message': 'No data available'}
        
        # Analyze performance by difficulty
        difficulty_performance = {}
        for qid, stats in self.question_stats.items():
            if stats['total_attempts'] > 0:
                difficulty_level = self._categorize_difficulty(stats['difficulty'])
                if difficulty_level not in difficulty_performance:
                    difficulty_performance[difficulty_level] = []
                difficulty_performance[difficulty_level].append(stats['success_rate'])
        
        # Calculate average performance by difficulty
        difficulty_analysis = {}
        for level, scores in difficulty_performance.items():
            difficulty_analysis[level] = {
                'average_success_rate': np.mean(scores),
                'question_count': len(scores),
                'recommendation': self._get_difficulty_recommendation(level, np.mean(scores))
            }
        
        # Overall learning insights
        total_questions = sum(stats['total_attempts'] for stats in self.question_stats.values())
        overall_success_rate = np.mean([
            stats['success_rate'] for stats in self.question_stats.values() 
            if stats['total_attempts'] > 0
        ])
        
        return {
            'total_questions_attempted': total_questions,
            'overall_success_rate': overall_success_rate,
            'difficulty_analysis': difficulty_analysis,
            'ability_level': self._categorize_ability(student_ability),
            'recommended_difficulty_range': self._get_recommended_difficulty_range(student_ability)
        }
    
    def _categorize_difficulty(self, difficulty: float) -> str:
        """Categorize difficulty level."""
        if difficulty < -1.0:
            return 'easy'
        elif difficulty < 1.0:
            return 'medium'
        else:
            return 'hard'
    
    def _categorize_ability(self, ability: float) -> str:
        """Categorize student ability level."""
        if ability < -1.0:
            return 'beginner'
        elif ability < 1.0:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _get_difficulty_recommendation(self, level: str, success_rate: float) -> str:
        """Get recommendation for difficulty level."""
        if success_rate > 0.8:
            return f"Student excels at {level} level questions"
        elif success_rate > 0.6:
            return f"Student performs well at {level} level questions"
        elif success_rate > 0.4:
            return f"Student struggles with {level} level questions"
        else:
            return f"Student needs more practice with {level} level questions"
    
    def _get_recommended_difficulty_range(self, ability: float) -> Tuple[float, float]:
        """Get recommended difficulty range for student."""
        center = ability
        width = 1.0  # ±1 difficulty units
        return (center - width, center + width)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            'question_stats': self.question_stats,
            'student_ability': self.student_ability,
            'exploration_factor': self.exploration_factor,
            'is_initialized': self.is_initialized
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.question_stats = model_data['question_stats']
        self.student_ability = model_data['student_ability']
        self.exploration_factor = model_data['exploration_factor']
        self.is_initialized = model_data['is_initialized']
