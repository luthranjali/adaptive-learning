"""Bayesian Knowledge Tracing (BKT) model implementation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
from app.config import settings


class BKTModel:
    """Bayesian Knowledge Tracing model for knowledge state estimation."""
    
    def __init__(self, initial_knowledge: float = None, learn_rate: float = None,
                 guess_rate: float = None, slip_rate: float = None):
        """Initialize BKT model with parameters."""
        self.initial_knowledge = initial_knowledge or settings.bkt_initial_knowledge
        self.learn_rate = learn_rate or settings.bkt_learn_rate
        self.guess_rate = guess_rate or settings.bkt_guess_rate
        self.slip_rate = slip_rate or settings.bkt_slip_rate
        
        # Skill-specific parameters (can be learned from data)
        self.skill_parameters = {}
        self.is_fitted = False
    
    def _logistic_function(self, x: float) -> float:
        """Sigmoid function for probability calculations."""
        return 1 / (1 + np.exp(-x))
    
    def _update_knowledge_state(self, current_knowledge: float, 
                               response: bool, skill_params: Dict) -> float:
        """Update knowledge state based on response."""
        p_learn = skill_params.get('learn_rate', self.learn_rate)
        p_guess = skill_params.get('guess_rate', self.guess_rate)
        p_slip = skill_params.get('slip_rate', self.slip_rate)
        
        if response:  # Correct response
            # P(knows|correct) = P(correct|knows) * P(knows) / P(correct)
            p_correct_given_knows = 1 - p_slip
            p_correct = p_correct_given_knows * current_knowledge + p_guess * (1 - current_knowledge)
            
            if p_correct > 0:
                new_knowledge = (p_correct_given_knows * current_knowledge) / p_correct
            else:
                new_knowledge = current_knowledge
        else:  # Incorrect response
            # P(knows|incorrect) = P(incorrect|knows) * P(knows) / P(incorrect)
            p_incorrect_given_knows = p_slip
            p_incorrect = p_incorrect_given_knows * current_knowledge + (1 - p_guess) * (1 - current_knowledge)
            
            if p_incorrect > 0:
                new_knowledge = (p_incorrect_given_knows * current_knowledge) / p_incorrect
            else:
                new_knowledge = current_knowledge
        
        # Apply learning rate
        if response and new_knowledge > current_knowledge:
            new_knowledge = current_knowledge + p_learn * (new_knowledge - current_knowledge)
        
        return np.clip(new_knowledge, 0.0, 1.0)
    
    def fit(self, skill_data: Dict[str, List[Tuple[bool, int]]]) -> Dict[str, any]:
        """
        Fit BKT model to skill-specific data.
        
        Args:
            skill_data: Dict mapping skill_id to list of (response, time) tuples
            
        Returns:
            Dictionary with fitted parameters and fit statistics
        """
        fit_results = {}
        
        for skill_id, responses in skill_data.items():
            if len(responses) < 5:  # Need minimum data
                self.skill_parameters[skill_id] = {
                    'learn_rate': self.learn_rate,
                    'guess_rate': self.guess_rate,
                    'slip_rate': self.slip_rate,
                    'initial_knowledge': self.initial_knowledge
                }
                continue
            
            # Extract response data
            response_values = [r[0] for r in responses]
            time_values = [r[1] for r in responses]
            
            # Simple parameter estimation (can be improved with EM algorithm)
            correct_responses = sum(response_values)
            total_responses = len(response_values)
            
            # Estimate guess rate from early incorrect responses
            early_responses = response_values[:max(1, len(response_values)//3)]
            guess_rate = max(0.01, min(0.5, 1 - sum(early_responses) / len(early_responses)))
            
            # Estimate slip rate from later correct responses
            later_responses = response_values[-max(1, len(response_values)//3):]
            slip_rate = max(0.01, min(0.5, 1 - sum(later_responses) / len(later_responses)))
            
            # Estimate learn rate from improvement over time
            learn_rate = self._estimate_learn_rate(response_values, time_values)
            
            self.skill_parameters[skill_id] = {
                'learn_rate': learn_rate,
                'guess_rate': guess_rate,
                'slip_rate': slip_rate,
                'initial_knowledge': self.initial_knowledge
            }
            
            fit_results[skill_id] = {
                'learn_rate': learn_rate,
                'guess_rate': guess_rate,
                'slip_rate': slip_rate,
                'total_responses': total_responses,
                'correct_responses': correct_responses
            }
        
        self.is_fitted = True
        return fit_results
    
    def _estimate_learn_rate(self, responses: List[bool], times: List[int]) -> float:
        """Estimate learning rate from response pattern."""
        if len(responses) < 3:
            return self.learn_rate
        
        # Simple heuristic: learning rate based on improvement over time
        early_performance = sum(responses[:len(responses)//2]) / (len(responses)//2)
        late_performance = sum(responses[len(responses)//2:]) / (len(responses) - len(responses)//2)
        
        improvement = late_performance - early_performance
        learn_rate = max(0.01, min(0.5, improvement))
        
        return learn_rate
    
    def predict_knowledge_state(self, skill_id: str, responses: List[bool]) -> List[float]:
        """Predict knowledge state progression for a skill."""
        if skill_id not in self.skill_parameters:
            # Use default parameters
            skill_params = {
                'learn_rate': self.learn_rate,
                'guess_rate': self.guess_rate,
                'slip_rate': self.slip_rate,
                'initial_knowledge': self.initial_knowledge
            }
        else:
            skill_params = self.skill_parameters[skill_id]
        
        knowledge_states = []
        current_knowledge = skill_params['initial_knowledge']
        
        for response in responses:
            knowledge_states.append(current_knowledge)
            current_knowledge = self._update_knowledge_state(
                current_knowledge, response, skill_params
            )
        
        return knowledge_states
    
    def predict_next_response(self, skill_id: str, current_knowledge: float) -> float:
        """Predict probability of correct response for next attempt."""
        if skill_id not in self.skill_parameters:
            skill_params = {
                'learn_rate': self.learn_rate,
                'guess_rate': self.guess_rate,
                'slip_rate': self.slip_rate
            }
        else:
            skill_params = self.skill_parameters[skill_id]
        
        p_guess = skill_params['guess_rate']
        p_slip = skill_params['slip_rate']
        
        # P(correct) = P(correct|knows) * P(knows) + P(correct|doesn't know) * P(doesn't know)
        p_correct = (1 - p_slip) * current_knowledge + p_guess * (1 - current_knowledge)
        
        return np.clip(p_correct, 0.0, 1.0)
    
    def get_skill_mastery(self, skill_id: str, responses: List[bool]) -> Dict[str, float]:
        """Get comprehensive skill mastery information."""
        knowledge_states = self.predict_knowledge_state(skill_id, responses)
        
        if not knowledge_states:
            return {
                'current_mastery': 0.0,
                'initial_mastery': 0.0,
                'learning_gain': 0.0,
                'mastery_trend': 0.0,
                'confidence': 0.0
            }
        
        current_mastery = knowledge_states[-1]
        initial_mastery = knowledge_states[0]
        learning_gain = current_mastery - initial_mastery
        
        # Calculate mastery trend (slope of knowledge states)
        if len(knowledge_states) > 1:
            x = np.arange(len(knowledge_states))
            slope = np.polyfit(x, knowledge_states, 1)[0]
            mastery_trend = slope
        else:
            mastery_trend = 0.0
        
        # Calculate confidence based on consistency
        if len(knowledge_states) > 2:
            variance = np.var(knowledge_states[-min(5, len(knowledge_states)):])
            confidence = max(0.0, 1.0 - variance)
        else:
            confidence = 0.5
        
        return {
            'current_mastery': current_mastery,
            'initial_mastery': initial_mastery,
            'learning_gain': learning_gain,
            'mastery_trend': mastery_trend,
            'confidence': confidence
        }
    
    def recommend_next_skill(self, skill_masteries: Dict[str, float], 
                            skill_dependencies: Dict[str, List[str]]) -> str:
        """Recommend next skill to focus on based on mastery levels and dependencies."""
        if not skill_masteries:
            return None
        
        # Find skills that are not yet mastered
        unmastered_skills = {
            skill: mastery for skill, mastery in skill_masteries.items() 
            if mastery < 0.8  # Mastery threshold
        }
        
        if not unmastered_skills:
            return None
        
        # Check prerequisites for each unmastered skill
        recommended_skills = []
        for skill, mastery in unmastered_skills.items():
            prerequisites = skill_dependencies.get(skill, [])
            prerequisites_met = all(
                skill_masteries.get(prereq, 0) >= 0.7  # Prerequisite threshold
                for prereq in prerequisites
            )
            
            if prerequisites_met:
                recommended_skills.append((skill, mastery))
        
        if not recommended_skills:
            # If no skills have met prerequisites, recommend the one with highest mastery
            return min(unmastered_skills.items(), key=lambda x: x[1])[0]
        
        # Recommend skill with lowest mastery among those with met prerequisites
        return min(recommended_skills, key=lambda x: x[1])[0]
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            'skill_parameters': self.skill_parameters,
            'initial_knowledge': self.initial_knowledge,
            'learn_rate': self.learn_rate,
            'guess_rate': self.guess_rate,
            'slip_rate': self.slip_rate,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.skill_parameters = model_data['skill_parameters']
        self.initial_knowledge = model_data['initial_knowledge']
        self.learn_rate = model_data['learn_rate']
        self.guess_rate = model_data['guess_rate']
        self.slip_rate = model_data['slip_rate']
        self.is_fitted = model_data['is_fitted']
