"""Item Response Theory (IRT) model implementation."""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import joblib
from app.config import settings


class IRTModel:
    """3-Parameter Logistic IRT model for question difficulty calibration."""
    
    def __init__(self):
        self.difficulty_params = {}  # b parameter
        self.discrimination_params = {}  # a parameter  
        self.guessing_params = {}  # c parameter
        self.is_fitted = False
        
    def _logistic_function(self, theta: float, a: float, b: float, c: float) -> float:
        """3PL logistic function."""
        return c + (1 - c) / (1 + np.exp(-a * (theta - b)))
    
    def _log_likelihood(self, params: np.ndarray, responses: np.ndarray, 
                       thetas: np.ndarray, question_idx: int) -> float:
        """Calculate log-likelihood for a single question."""
        a, b, c = params
        
        # Ensure parameters are within valid ranges
        a = max(0.1, min(2.0, a))
        c = max(0.0, min(0.3, c))
        
        probabilities = self._logistic_function(thetas, a, b, c)
        
        # Avoid log(0) by adding small epsilon
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
        
        log_likelihood = np.sum(
            responses * np.log(probabilities) + 
            (1 - responses) * np.log(1 - probabilities)
        )
        
        return -log_likelihood  # Minimize negative log-likelihood
    
    def fit(self, responses: np.ndarray, thetas: np.ndarray, 
            question_ids: List[int]) -> Dict[str, any]:
        """
        Fit IRT model to response data.
        
        Args:
            responses: Binary response matrix (students x questions)
            thetas: Student ability estimates
            question_ids: List of question IDs
            
        Returns:
            Dictionary with fitted parameters and fit statistics
        """
        n_questions = len(question_ids)
        fit_results = {}
        
        for i, qid in enumerate(question_ids):
            question_responses = responses[:, i]
            
            # Initial parameter estimates
            initial_params = [1.0, 0.0, 0.1]  # [a, b, c]
            
            # Bounds for parameters
            bounds = [
                (0.1, 2.0),  # discrimination
                (-3.0, 3.0),  # difficulty
                (0.0, 0.3)   # guessing
            ]
            
            try:
                result = minimize(
                    self._log_likelihood,
                    initial_params,
                    args=(question_responses, thetas, i),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    a, b, c = result.x
                    self.difficulty_params[qid] = b
                    self.discrimination_params[qid] = a
                    self.guessing_params[qid] = c
                    
                    fit_results[qid] = {
                        'difficulty': b,
                        'discrimination': a,
                        'guessing': c,
                        'log_likelihood': -result.fun,
                        'converged': True
                    }
                else:
                    # Use default parameters if optimization fails
                    self.difficulty_params[qid] = 0.0
                    self.discrimination_params[qid] = 1.0
                    self.guessing_params[qid] = 0.1
                    
                    fit_results[qid] = {
                        'difficulty': 0.0,
                        'discrimination': 1.0,
                        'guessing': 0.1,
                        'log_likelihood': float('inf'),
                        'converged': False
                    }
                    
            except Exception as e:
                print(f"Error fitting question {qid}: {e}")
                # Use default parameters
                self.difficulty_params[qid] = 0.0
                self.discrimination_params[qid] = 1.0
                self.guessing_params[qid] = 0.1
                
                fit_results[qid] = {
                    'difficulty': 0.0,
                    'discrimination': 1.0,
                    'guessing': 0.1,
                    'log_likelihood': float('inf'),
                    'converged': False,
                    'error': str(e)
                }
        
        self.is_fitted = True
        return fit_results
    
    def predict_probability(self, theta: float, question_id: int) -> float:
        """Predict probability of correct response for given ability and question."""
        if not self.is_fitted or question_id not in self.difficulty_params:
            return 0.5  # Default probability
        
        a = self.discrimination_params[question_id]
        b = self.difficulty_params[question_id]
        c = self.guessing_params[question_id]
        
        return self._logistic_function(theta, a, b, c)
    
    def get_question_info(self, question_id: int) -> Dict[str, float]:
        """Get IRT parameters for a question."""
        if not self.is_fitted or question_id not in self.difficulty_params:
            return {'difficulty': 0.0, 'discrimination': 1.0, 'guessing': 0.1}
        
        return {
            'difficulty': self.difficulty_params[question_id],
            'discrimination': self.discrimination_params[question_id],
            'guessing': self.guessing_params[question_id]
        }
    
    def estimate_ability(self, responses: Dict[int, bool], 
                        question_ids: List[int]) -> float:
        """Estimate student ability from response pattern."""
        if not self.is_fitted:
            return 0.0
        
        def ability_log_likelihood(theta):
            log_likelihood = 0.0
            for qid in question_ids:
                if qid in responses:
                    prob = self.predict_probability(theta, qid)
                    prob = np.clip(prob, 1e-10, 1 - 1e-10)
                    
                    if responses[qid]:
                        log_likelihood += np.log(prob)
                    else:
                        log_likelihood += np.log(1 - prob)
            return -log_likelihood
        
        # Find ability that maximizes likelihood
        result = minimize(
            ability_log_likelihood,
            [0.0],
            bounds=[(-3.0, 3.0)],
            method='L-BFGS-B'
        )
        
        return result.x[0] if result.success else 0.0
    
    def get_optimal_difficulty(self, current_ability: float, 
                             target_probability: float = 0.5) -> float:
        """Get optimal question difficulty for target probability."""
        if not self.is_fitted:
            return current_ability
        
        # For 3PL model: b = theta - (1/a) * ln((1-c)/(p-c) - 1)
        # Using average discrimination and guessing parameters
        avg_discrimination = np.mean(list(self.discrimination_params.values())) if self.discrimination_params else 1.0
        avg_guessing = np.mean(list(self.guessing_params.values())) if self.guessing_params else 0.1
        
        if avg_discrimination == 0 or target_probability <= avg_guessing:
            return current_ability
        
        optimal_difficulty = current_ability - (1/avg_discrimination) * np.log(
            (1 - avg_guessing) / (target_probability - avg_guessing) - 1
        )
        
        return np.clip(optimal_difficulty, -3.0, 3.0)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            'difficulty_params': self.difficulty_params,
            'discrimination_params': self.discrimination_params,
            'guessing_params': self.guessing_params,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.difficulty_params = model_data['difficulty_params']
        self.discrimination_params = model_data['discrimination_params']
        self.guessing_params = model_data['guessing_params']
        self.is_fitted = model_data['is_fitted']
