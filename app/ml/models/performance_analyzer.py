"""Performance analysis model using ensemble methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """Ensemble-based performance analysis model."""
    
    def __init__(self, model_type: str = 'ensemble'):
        """Initialize performance analyzer."""
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = {}
        
        # Initialize models based on type
        if model_type == 'ensemble':
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
        elif model_type == 'random_forest':
            self.models = {'main': RandomForestRegressor(n_estimators=100, random_state=42)}
        elif model_type == 'gradient_boosting':
            self.models = {'main': GradientBoostingRegressor(n_estimators=100, random_state=42)}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _extract_performance_features(self, assessment_data: Dict) -> np.ndarray:
        """Extract features for performance analysis."""
        features = []
        
        # Basic performance metrics
        features.extend([
            assessment_data.get('total_score', 0.0),
            assessment_data.get('percentage_score', 0.0),
            assessment_data.get('correct_answers', 0),
            assessment_data.get('total_questions', 1),
            assessment_data.get('time_spent_seconds', 0),
            assessment_data.get('average_time_per_question', 0.0)
        ])
        
        # Time-based features
        time_features = assessment_data.get('time_features', {})
        features.extend([
            time_features.get('time_consistency', 0.0),
            time_features.get('speed_accuracy_tradeoff', 0.0),
            time_features.get('time_management_score', 0.0),
            time_features.get('response_time_variance', 0.0)
        ])
        
        # Difficulty progression
        difficulty_progression = assessment_data.get('difficulty_progression', [])
        if difficulty_progression:
            features.extend([
                np.mean(difficulty_progression),
                np.std(difficulty_progression),
                np.max(difficulty_progression),
                np.min(difficulty_progression),
                np.median(difficulty_progression),
                len(difficulty_progression)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0])
        
        # Learning patterns
        learning_patterns = assessment_data.get('learning_patterns', {})
        features.extend([
            learning_patterns.get('consistency_score', 0.0),
            learning_patterns.get('improvement_rate', 0.0),
            learning_patterns.get('retention_rate', 0.0),
            learning_patterns.get('engagement_score', 0.0),
            learning_patterns.get('persistence_score', 0.0)
        ])
        
        # Knowledge state features
        knowledge_state = assessment_data.get('knowledge_state', {})
        features.extend([
            knowledge_state.get('overall_mastery', 0.0),
            knowledge_state.get('skill_count', 0),
            knowledge_state.get('mastered_skills', 0),
            knowledge_state.get('learning_velocity', 0.0),
            knowledge_state.get('confidence_level', 0.0)
        ])
        
        # Error analysis
        error_patterns = assessment_data.get('error_patterns', {})
        features.extend([
            error_patterns.get('conceptual_errors', 0),
            error_patterns.get('procedural_errors', 0),
            error_patterns.get('careless_errors', 0),
            error_patterns.get('total_errors', 0),
            error_patterns.get('error_rate', 0.0)
        ])
        
        # Subject-specific performance
        subject_scores = assessment_data.get('subject_scores', {})
        for subject in ['math', 'science', 'english', 'history', 'reading', 'writing']:
            features.append(subject_scores.get(subject, 0.0))
        
        # Interaction patterns
        interaction_patterns = assessment_data.get('interaction_patterns', {})
        features.extend([
            interaction_patterns.get('hint_usage_rate', 0.0),
            interaction_patterns.get('retry_rate', 0.0),
            interaction_patterns.get('skip_rate', 0.0),
            interaction_patterns.get('help_seeking_frequency', 0.0)
        ])
        
        # Student profile features
        student_profile = assessment_data.get('student_profile', {})
        features.extend([
            student_profile.get('grade_level', 0),
            student_profile.get('previous_performance', 0.0),
            student_profile.get('learning_style_score', 0.0),
            student_profile.get('motivation_score', 0.0),
            student_profile.get('self_efficacy', 0.0)
        ])
        
        return np.array(features)
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for performance analysis."""
        X = []
        y = []
        
        for record in training_data:
            features = self._extract_performance_features(record['assessment_data'])
            X.append(features)
            
            # Target variable (could be various performance metrics)
            target = record.get('target_performance', record['assessment_data'].get('percentage_score', 0.0))
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0)
        
        return X, y
    
    def fit(self, training_data: List[Dict], validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit performance analysis models.
        
        Args:
            training_data: List of training records
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training results
        """
        if not training_data:
            return {'error': 'No training data provided'}
        
        X, y = self._prepare_training_data(training_data)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train models
        model_results = {}
        
        for model_name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            model_results[model_name] = {
                'mse': mse,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_.tolist()
        
        # For ensemble, create meta-model
        if self.model_type == 'ensemble':
            self._create_meta_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        self.is_fitted = True
        
        return {
            'model_results': model_results,
            'feature_importance': self.feature_importance,
            'n_features': X.shape[1],
            'n_samples': len(X)
        }
    
    def _create_meta_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Create meta-model for ensemble predictions."""
        # Generate base model predictions
        train_predictions = []
        val_predictions = []
        
        for model_name, model in self.models.items():
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_predictions.append(train_pred)
            val_predictions.append(val_pred)
        
        # Stack predictions
        X_train_meta = np.column_stack(train_predictions)
        X_val_meta = np.column_stack(val_predictions)
        
        # Train meta-model
        meta_model = LinearRegression()
        meta_model.fit(X_train_meta, y_train)
        
        # Store meta-model
        self.models['meta'] = meta_model
        
        # Evaluate meta-model
        y_pred_meta = meta_model.predict(X_val_meta)
        mse_meta = mean_squared_error(y_val, y_pred_meta)
        r2_meta = r2_score(y_val, y_pred_meta)
        
        print(f"Meta-model performance - MSE: {mse_meta:.4f}, R²: {r2_meta:.4f}")
    
    def predict_performance(self, assessment_data: Dict) -> Dict[str, Any]:
        """Predict performance metrics for assessment data."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        features = self._extract_performance_features(assessment_data)
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        predictions = {}
        
        if self.model_type == 'ensemble':
            # Get base model predictions
            base_predictions = []
            for model_name, model in self.models.items():
                if model_name != 'meta':
                    pred = model.predict(features_scaled)[0]
                    predictions[f'{model_name}_prediction'] = float(pred)
                    base_predictions.append(pred)
            
            # Get meta-model prediction
            if 'meta' in self.models:
                meta_pred = self.models['meta'].predict([base_predictions])[0]
                predictions['ensemble_prediction'] = float(meta_pred)
                predictions['final_prediction'] = float(meta_pred)
            else:
                predictions['final_prediction'] = float(np.mean(base_predictions))
        else:
            # Single model prediction
            model_name = list(self.models.keys())[0]
            pred = self.models[model_name].predict(features_scaled)[0]
            predictions['prediction'] = float(pred)
            predictions['final_prediction'] = float(pred)
        
        # Add confidence estimation
        predictions['confidence'] = self._estimate_confidence(features_scaled)
        
        # Add performance insights
        predictions['insights'] = self._generate_performance_insights(assessment_data, predictions['final_prediction'])
        
        return predictions
    
    def _estimate_confidence(self, features: np.ndarray) -> float:
        """Estimate prediction confidence based on feature similarity to training data."""
        # Simple confidence estimation based on feature variance
        # In practice, this could be more sophisticated
        feature_variance = np.var(features)
        confidence = max(0.0, min(1.0, 1.0 - feature_variance))
        return confidence
    
    def _generate_performance_insights(self, assessment_data: Dict, predicted_score: float) -> List[str]:
        """Generate insights about predicted performance."""
        insights = []
        
        # Score-based insights
        if predicted_score >= 90:
            insights.append("Predicted excellent performance")
        elif predicted_score >= 80:
            insights.append("Predicted strong performance")
        elif predicted_score >= 70:
            insights.append("Predicted good performance")
        elif predicted_score >= 60:
            insights.append("Predicted satisfactory performance")
        else:
            insights.append("Predicted performance needs improvement")
        
        # Time-based insights
        time_spent = assessment_data.get('time_spent_seconds', 0)
        total_questions = assessment_data.get('total_questions', 1)
        avg_time = time_spent / total_questions if total_questions > 0 else 0
        
        if avg_time < 30:
            insights.append("Very fast response times - consider accuracy")
        elif avg_time > 120:
            insights.append("Slow response times - consider time management")
        else:
            insights.append("Appropriate response times")
        
        # Consistency insights
        consistency_score = assessment_data.get('learning_patterns', {}).get('consistency_score', 0)
        if consistency_score > 0.8:
            insights.append("High consistency in performance")
        elif consistency_score < 0.4:
            insights.append("Inconsistent performance - focus on stability")
        
        # Knowledge state insights
        overall_mastery = assessment_data.get('knowledge_state', {}).get('overall_mastery', 0)
        if overall_mastery > 0.8:
            insights.append("Strong conceptual understanding")
        elif overall_mastery < 0.4:
            insights.append("Conceptual understanding needs development")
        
        return insights
    
    def analyze_learning_patterns(self, assessment_data: Dict) -> Dict[str, Any]:
        """Analyze learning patterns from assessment data."""
        patterns = {}
        
        # Performance progression
        difficulty_progression = assessment_data.get('difficulty_progression', [])
        if len(difficulty_progression) > 1:
            progression_trend = np.polyfit(range(len(difficulty_progression)), difficulty_progression, 1)[0]
            patterns['difficulty_trend'] = {
                'slope': float(progression_trend),
                'trend': 'improving' if progression_trend > 0 else 'declining' if progression_trend < 0 else 'stable'
            }
        
        # Time patterns
        time_features = assessment_data.get('time_features', {})
        patterns['time_analysis'] = {
            'consistency': time_features.get('time_consistency', 0.0),
            'speed_accuracy_tradeoff': time_features.get('speed_accuracy_tradeoff', 0.0),
            'management_score': time_features.get('time_management_score', 0.0)
        }
        
        # Learning behavior
        interaction_patterns = assessment_data.get('interaction_patterns', {})
        patterns['learning_behavior'] = {
            'hint_dependency': interaction_patterns.get('hint_usage_rate', 0.0),
            'persistence': 1.0 - interaction_patterns.get('skip_rate', 0.0),
            'help_seeking': interaction_patterns.get('help_seeking_frequency', 0.0)
        }
        
        # Error patterns
        error_patterns = assessment_data.get('error_patterns', {})
        total_errors = error_patterns.get('total_errors', 0)
        if total_errors > 0:
            patterns['error_analysis'] = {
                'conceptual_error_rate': error_patterns.get('conceptual_errors', 0) / total_errors,
                'procedural_error_rate': error_patterns.get('procedural_errors', 0) / total_errors,
                'careless_error_rate': error_patterns.get('careless_errors', 0) / total_errors
            }
        
        return patterns
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """Get feature importance from models."""
        return self.feature_importance
    
    def predict_batch(self, assessment_data_list: List[Dict]) -> List[Dict[str, Any]]:
        """Predict performance for multiple assessments."""
        if not self.is_fitted:
            return [{'error': 'Model not trained'} for _ in assessment_data_list]
        
        results = []
        for assessment_data in assessment_data_list:
            prediction = self.predict_performance(assessment_data)
            results.append(prediction)
        
        return results
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']
        self.feature_importance = model_data['feature_importance']
