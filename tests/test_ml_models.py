"""Tests for ML models."""

import pytest
import numpy as np
from app.ml.models.irt_model import IRTModel
from app.ml.models.bkt_model import BKTModel
from app.ml.models.level_recommender import LevelRecommender
from app.ml.models.question_selector import QuestionSelector
from app.ml.models.gap_detector import GapDetector
from app.ml.models.performance_analyzer import PerformanceAnalyzer


class TestIRTModel:
    """Test IRT model functionality."""
    
    def test_irt_model_initialization(self):
        """Test IRT model initialization."""
        model = IRTModel()
        assert not model.is_fitted
        assert model.difficulty_params == {}
        assert model.discrimination_params == {}
        assert model.guessing_params == {}
    
    def test_predict_probability(self):
        """Test probability prediction."""
        model = IRTModel()
        model.difficulty_params = {1: 0.0}
        model.discrimination_params = {1: 1.0}
        model.guessing_params = {1: 0.1}
        model.is_fitted = True
        
        prob = model.predict_probability(0.0, 1)
        assert 0.0 <= prob <= 1.0
    
    def test_estimate_ability(self):
        """Test ability estimation."""
        model = IRTModel()
        model.difficulty_params = {1: 0.0, 2: 1.0}
        model.discrimination_params = {1: 1.0, 2: 1.0}
        model.guessing_params = {1: 0.1, 2: 0.1}
        model.is_fitted = True
        
        responses = {1: True, 2: False}
        ability = model.estimate_ability(responses, [1, 2])
        assert isinstance(ability, float)


class TestBKTModel:
    """Test BKT model functionality."""
    
    def test_bkt_model_initialization(self):
        """Test BKT model initialization."""
        model = BKTModel()
        assert not model.is_fitted
        assert model.initial_knowledge == 0.1
        assert model.learn_rate == 0.3
    
    def test_predict_knowledge_state(self):
        """Test knowledge state prediction."""
        model = BKTModel()
        responses = [True, False, True, True]
        knowledge_states = model.predict_knowledge_state("test_skill", responses)
        
        assert len(knowledge_states) == len(responses)
        assert all(0.0 <= state <= 1.0 for state in knowledge_states)
    
    def test_get_skill_mastery(self):
        """Test skill mastery analysis."""
        model = BKTModel()
        responses = [True, False, True, True, True]
        mastery = model.get_skill_mastery("test_skill", responses)
        
        assert 'current_mastery' in mastery
        assert 'learning_gain' in mastery
        assert 'confidence' in mastery


class TestLevelRecommender:
    """Test level recommender functionality."""
    
    def test_level_recommender_initialization(self):
        """Test level recommender initialization."""
        model = LevelRecommender()
        assert not model.is_fitted
        assert model.model is not None
    
    def test_extract_features(self):
        """Test feature extraction."""
        model = LevelRecommender()
        assessment_data = {
            'total_score': 85.0,
            'percentage_score': 85.0,
            'correct_answers': 17,
            'total_questions': 20,
            'time_spent_seconds': 1200,
            'average_time_per_question': 60.0
        }
        
        features = model._extract_features(assessment_data)
        assert len(features) > 0
        assert isinstance(features, np.ndarray)


class TestQuestionSelector:
    """Test question selector functionality."""
    
    def test_question_selector_initialization(self):
        """Test question selector initialization."""
        selector = QuestionSelector()
        assert selector.exploration_factor == 0.1
        assert not selector.is_initialized
    
    def test_select_next_question(self):
        """Test question selection."""
        selector = QuestionSelector()
        available_questions = [
            {'id': 1, 'difficulty': 0.0, 'subject': 'math', 'topic': 'algebra'},
            {'id': 2, 'difficulty': 1.0, 'subject': 'math', 'topic': 'geometry'}
        ]
        
        student_ability = 0.5
        assessment_context = {'target_subjects': ['math']}
        
        selected = selector.select_next_question(
            available_questions, student_ability, assessment_context
        )
        
        assert selected is not None
        assert 'id' in selected
        assert 'selection_utility' in selected


class TestGapDetector:
    """Test gap detector functionality."""
    
    def test_gap_detector_initialization(self):
        """Test gap detector initialization."""
        detector = GapDetector()
        assert detector.clustering_method == 'kmeans'
        assert detector.n_clusters == 3
        assert not detector.is_fitted
    
    def test_extract_skill_features(self):
        """Test skill feature extraction."""
        detector = GapDetector()
        student_data = {
            'overall_score': 75.0,
            'completion_rate': 0.9,
            'subject_scores': {'math': 0.8, 'science': 0.7},
            'skill_mastery': {'problem_solving': 0.6, 'critical_thinking': 0.5}
        }
        
        features = detector._extract_skill_features(student_data)
        assert len(features) > 0
        assert isinstance(features, np.ndarray)


class TestPerformanceAnalyzer:
    """Test performance analyzer functionality."""
    
    def test_performance_analyzer_initialization(self):
        """Test performance analyzer initialization."""
        analyzer = PerformanceAnalyzer()
        assert analyzer.model_type == 'ensemble'
        assert not analyzer.is_fitted
        assert len(analyzer.models) > 0
    
    def test_extract_performance_features(self):
        """Test performance feature extraction."""
        analyzer = PerformanceAnalyzer()
        assessment_data = {
            'total_score': 80.0,
            'percentage_score': 80.0,
            'time_spent_seconds': 1800,
            'learning_patterns': {'consistency_score': 0.8}
        }
        
        features = analyzer._extract_performance_features(assessment_data)
        assert len(features) > 0
        assert isinstance(features, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])
