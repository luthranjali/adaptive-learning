"""Level recommendation model using XGBoost."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class LevelRecommender:
    """XGBoost-based model for course level recommendations."""
    
    def __init__(self, model_params: Dict = None):
        """Initialize level recommender with XGBoost."""
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        if model_params:
            self.model.set_params(**model_params)
        
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False
        self.feature_importance = {}
        
    def _extract_features(self, assessment_data: Dict) -> np.ndarray:
        """Extract features from assessment data."""
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
        
        # Subject-specific scores
        subject_scores = assessment_data.get('subject_scores', {})
        for subject in ['math', 'science', 'english', 'history']:
            features.append(subject_scores.get(subject, 0.0))
        
        # Difficulty progression
        difficulty_progression = assessment_data.get('difficulty_progression', [])
        if difficulty_progression:
            features.extend([
                np.mean(difficulty_progression),
                np.std(difficulty_progression),
                np.max(difficulty_progression),
                np.min(difficulty_progression)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Learning patterns
        learning_patterns = assessment_data.get('learning_patterns', {})
        features.extend([
            learning_patterns.get('consistency_score', 0.0),
            learning_patterns.get('improvement_rate', 0.0),
            learning_patterns.get('retention_rate', 0.0),
            learning_patterns.get('engagement_score', 0.0)
        ])
        
        # Knowledge state (BKT results)
        knowledge_state = assessment_data.get('knowledge_state', {})
        if knowledge_state:
            features.extend([
                knowledge_state.get('overall_mastery', 0.0),
                knowledge_state.get('skill_count', 0),
                knowledge_state.get('mastered_skills', 0),
                knowledge_state.get('learning_velocity', 0.0)
            ])
        else:
            features.extend([0.0, 0, 0, 0.0])
        
        # Error patterns
        error_patterns = assessment_data.get('error_patterns', {})
        features.extend([
            error_patterns.get('conceptual_errors', 0),
            error_patterns.get('procedural_errors', 0),
            error_patterns.get('careless_errors', 0),
            error_patterns.get('total_errors', 0)
        ])
        
        # Time-based features
        time_features = assessment_data.get('time_features', {})
        features.extend([
            time_features.get('time_consistency', 0.0),
            time_features.get('speed_accuracy_tradeoff', 0.0),
            time_features.get('time_management_score', 0.0)
        ])
        
        # Prerequisite knowledge
        prerequisite_scores = assessment_data.get('prerequisite_scores', {})
        features.extend([
            prerequisite_scores.get('basic_skills', 0.0),
            prerequisite_scores.get('foundational_concepts', 0.0),
            prerequisite_scores.get('advanced_concepts', 0.0)
        ])
        
        # Student profile features
        student_profile = assessment_data.get('student_profile', {})
        features.extend([
            student_profile.get('grade_level', 0),
            student_profile.get('previous_course_level', 0),
            student_profile.get('learning_style_score', 0.0),
            student_profile.get('motivation_score', 0.0)
        ])
        
        return np.array(features)
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from assessment records."""
        X = []
        y = []
        
        for record in training_data:
            features = self._extract_features(record['assessment_data'])
            X.append(features)
            y.append(record['actual_level'])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def fit(self, training_data: List[Dict], validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit the level recommendation model.
        
        Args:
            training_data: List of training records with 'assessment_data' and 'actual_level'
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training results and metrics
        """
        X, y = self._prepare_training_data(training_data)
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_val)
        
        # Feature importance
        self.feature_importance = dict(zip(
            range(len(self.feature_columns) if self.feature_columns else range(X.shape[1])),
            self.model.feature_importances_
        ))
        
        self.is_fitted = True
        
        return {
            'accuracy': accuracy,
            'feature_importance': self.feature_importance,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'classes': self.label_encoder.classes_.tolist()
        }
    
    def predict_level(self, assessment_data: Dict) -> Dict[str, Any]:
        """Predict recommended course level."""
        if not self.is_fitted:
            return {
                'recommended_level': 'intermediate',
                'confidence': 0.5,
                'reasoning': 'Model not trained'
            }
        
        features = self._extract_features(assessment_data)
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        prediction_proba = self.model.predict_proba(features)[0]
        
        # Get confidence
        confidence = np.max(prediction_proba)
        
        # Decode prediction
        recommended_level = self.label_encoder.inverse_transform([prediction])[0]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(assessment_data, recommended_level, confidence)
        
        return {
            'recommended_level': recommended_level,
            'confidence': float(confidence),
            'reasoning': reasoning,
            'all_probabilities': dict(zip(
                self.label_encoder.classes_,
                prediction_proba.tolist()
            ))
        }
    
    def _generate_reasoning(self, assessment_data: Dict, level: str, confidence: float) -> str:
        """Generate human-readable reasoning for the recommendation."""
        reasoning_parts = []
        
        # Performance-based reasoning
        percentage_score = assessment_data.get('percentage_score', 0)
        if percentage_score >= 90:
            reasoning_parts.append(f"Excellent performance ({percentage_score:.1f}%)")
        elif percentage_score >= 80:
            reasoning_parts.append(f"Strong performance ({percentage_score:.1f}%)")
        elif percentage_score >= 70:
            reasoning_parts.append(f"Good performance ({percentage_score:.1f}%)")
        else:
            reasoning_parts.append(f"Performance needs improvement ({percentage_score:.1f}%)")
        
        # Knowledge state reasoning
        knowledge_state = assessment_data.get('knowledge_state', {})
        overall_mastery = knowledge_state.get('overall_mastery', 0)
        if overall_mastery >= 0.8:
            reasoning_parts.append("Strong conceptual understanding")
        elif overall_mastery >= 0.6:
            reasoning_parts.append("Good conceptual understanding")
        else:
            reasoning_parts.append("Conceptual understanding needs development")
        
        # Learning pattern reasoning
        learning_patterns = assessment_data.get('learning_patterns', {})
        improvement_rate = learning_patterns.get('improvement_rate', 0)
        if improvement_rate > 0.1:
            reasoning_parts.append("Showing strong learning progress")
        elif improvement_rate > 0.05:
            reasoning_parts.append("Showing steady learning progress")
        else:
            reasoning_parts.append("Learning progress could be improved")
        
        # Confidence-based reasoning
        if confidence >= 0.8:
            reasoning_parts.append("High confidence in recommendation")
        elif confidence >= 0.6:
            reasoning_parts.append("Moderate confidence in recommendation")
        else:
            reasoning_parts.append("Lower confidence - consider additional assessment")
        
        return ". ".join(reasoning_parts) + "."
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            return {}
        
        # Map feature indices to names if available
        if self.feature_columns:
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            return {f"feature_{i}": importance for i, importance in enumerate(self.model.feature_importances_)}
    
    def predict_batch(self, assessment_data_list: List[Dict]) -> List[Dict[str, Any]]:
        """Predict levels for multiple assessments."""
        if not self.is_fitted:
            return [{'error': 'Model not trained'} for _ in assessment_data_list]
        
        X = np.array([self._extract_features(data) for data in assessment_data_list])
        
        predictions = self.model.predict(X)
        prediction_probas = self.model.predict_proba(X)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
            recommended_level = self.label_encoder.inverse_transform([pred])[0]
            confidence = np.max(proba)
            
            results.append({
                'recommended_level': recommended_level,
                'confidence': float(confidence),
                'all_probabilities': dict(zip(
                    self.label_encoder.classes_,
                    proba.tolist()
                ))
            })
        
        return results
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']
        self.feature_importance = model_data['feature_importance']
