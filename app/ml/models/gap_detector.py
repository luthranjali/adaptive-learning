"""Knowledge gap detection using clustering and anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class GapDetector:
    """Knowledge gap detection using clustering and anomaly detection."""
    
    def __init__(self, clustering_method: str = 'kmeans', n_clusters: int = 3):
        """Initialize gap detector."""
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.clusterer = None
        self.anomaly_detector = None
        self.is_fitted = False
        self.skill_clusters = {}
        self.gap_patterns = {}
        
    def _extract_skill_features(self, student_data: Dict) -> np.ndarray:
        """Extract features for skill gap analysis."""
        features = []
        
        # Performance metrics
        features.extend([
            student_data.get('overall_score', 0.0),
            student_data.get('completion_rate', 0.0),
            student_data.get('average_time_per_question', 0.0),
            student_data.get('consistency_score', 0.0)
        ])
        
        # Subject-specific performance
        subject_scores = student_data.get('subject_scores', {})
        for subject in ['math', 'science', 'english', 'history', 'reading', 'writing']:
            features.append(subject_scores.get(subject, 0.0))
        
        # Skill mastery levels
        skill_mastery = student_data.get('skill_mastery', {})
        mastery_scores = []
        for skill in ['problem_solving', 'critical_thinking', 'analytical_reasoning', 
                     'memory_recall', 'pattern_recognition', 'logical_reasoning']:
            mastery_scores.append(skill_mastery.get(skill, 0.0))
        features.extend(mastery_scores)
        
        # Error patterns
        error_patterns = student_data.get('error_patterns', {})
        features.extend([
            error_patterns.get('conceptual_errors', 0),
            error_patterns.get('procedural_errors', 0),
            error_patterns.get('careless_errors', 0),
            error_patterns.get('time_pressure_errors', 0)
        ])
        
        # Learning behavior
        learning_behavior = student_data.get('learning_behavior', {})
        features.extend([
            learning_behavior.get('hint_usage_rate', 0.0),
            learning_behavior.get('retry_rate', 0.0),
            learning_behavior.get('skip_rate', 0.0),
            learning_behavior.get('help_seeking_frequency', 0.0)
        ])
        
        # Time-based patterns
        time_patterns = student_data.get('time_patterns', {})
        features.extend([
            time_patterns.get('time_consistency', 0.0),
            time_patterns.get('speed_accuracy_tradeoff', 0.0),
            time_patterns.get('time_management_score', 0.0)
        ])
        
        # Difficulty progression
        difficulty_progression = student_data.get('difficulty_progression', [])
        if difficulty_progression:
            features.extend([
                np.mean(difficulty_progression),
                np.std(difficulty_progression),
                np.max(difficulty_progression) - np.min(difficulty_progression)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _prepare_clustering_data(self, student_data_list: List[Dict]) -> np.ndarray:
        """Prepare data for clustering analysis."""
        X = []
        for student_data in student_data_list:
            features = self._extract_skill_features(student_data)
            X.append(features)
        
        X = np.array(X)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        return X
    
    def fit(self, student_data_list: List[Dict], skill_categories: List[str] = None) -> Dict[str, Any]:
        """
        Fit gap detection models.
        
        Args:
            student_data_list: List of student performance data
            skill_categories: List of skill categories to analyze
            
        Returns:
            Dictionary with fit results and gap analysis
        """
        if not student_data_list:
            return {'error': 'No data provided'}
        
        X = self._prepare_clustering_data(student_data_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters
        if self.clustering_method == 'kmeans':
            optimal_clusters = self._find_optimal_clusters(X_scaled)
            self.n_clusters = optimal_clusters
            
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.clustering_method == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Fit clustering model
        cluster_labels = self.clusterer.fit_predict(X_scaled)
        
        # Fit anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        anomaly_labels = self.anomaly_detector.fit_predict(X_scaled)
        
        # Analyze clusters for gap patterns
        gap_analysis = self._analyze_gap_patterns(X_scaled, cluster_labels, skill_categories)
        
        # Identify skill-specific gaps
        skill_gaps = self._identify_skill_gaps(student_data_list, cluster_labels)
        
        self.is_fitted = True
        
        return {
            'n_clusters': self.n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'anomaly_labels': anomaly_labels.tolist(),
            'gap_analysis': gap_analysis,
            'skill_gaps': skill_gaps,
            'silhouette_score': silhouette_score(X_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
        }
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using silhouette score."""
        if len(X) < 2:
            return 1
        
        best_score = -1
        best_k = 2
        
        for k in range(2, min(max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(X, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        return best_k
    
    def _analyze_gap_patterns(self, X: np.ndarray, cluster_labels: np.ndarray, 
                            skill_categories: List[str] = None) -> Dict[str, Any]:
        """Analyze patterns in clusters to identify gaps."""
        gap_patterns = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate cluster characteristics
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)
            
            # Identify low-performing features (potential gaps)
            low_performance_threshold = 0.3
            gap_features = np.where(cluster_mean < low_performance_threshold)[0]
            
            gap_patterns[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'mean_performance': cluster_mean.tolist(),
                'gap_features': gap_features.tolist(),
                'gap_severity': np.mean(cluster_mean[gap_features]) if len(gap_features) > 0 else 0.0,
                'characteristics': self._describe_cluster_characteristics(cluster_mean, skill_categories)
            }
        
        return gap_patterns
    
    def _describe_cluster_characteristics(self, cluster_mean: np.ndarray, 
                                        skill_categories: List[str] = None) -> Dict[str, str]:
        """Describe characteristics of a cluster."""
        characteristics = {}
        
        # Performance level
        overall_performance = np.mean(cluster_mean[:4])  # First 4 features are performance metrics
        if overall_performance > 0.7:
            characteristics['performance_level'] = 'High'
        elif overall_performance > 0.4:
            characteristics['performance_level'] = 'Medium'
        else:
            characteristics['performance_level'] = 'Low'
        
        # Subject strengths/weaknesses
        subject_scores = cluster_mean[4:10]  # Subject scores
        subject_names = ['math', 'science', 'english', 'history', 'reading', 'writing']
        
        strengths = []
        weaknesses = []
        for i, score in enumerate(subject_scores):
            if i < len(subject_names):
                if score > 0.6:
                    strengths.append(subject_names[i])
                elif score < 0.4:
                    weaknesses.append(subject_names[i])
        
        characteristics['subject_strengths'] = strengths
        characteristics['subject_weaknesses'] = weaknesses
        
        # Skill patterns
        skill_scores = cluster_mean[10:16]  # Skill mastery scores
        skill_names = ['problem_solving', 'critical_thinking', 'analytical_reasoning',
                      'memory_recall', 'pattern_recognition', 'logical_reasoning']
        
        strong_skills = []
        weak_skills = []
        for i, score in enumerate(skill_scores):
            if i < len(skill_names):
                if score > 0.6:
                    strong_skills.append(skill_names[i])
                elif score < 0.4:
                    weak_skills.append(skill_names[i])
        
        characteristics['strong_skills'] = strong_skills
        characteristics['weak_skills'] = weak_skills
        
        return characteristics
    
    def _identify_skill_gaps(self, student_data_list: List[Dict], 
                           cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Identify specific skill gaps across students."""
        skill_gaps = {}
        
        # Analyze each skill category
        skill_categories = ['problem_solving', 'critical_thinking', 'analytical_reasoning',
                           'memory_recall', 'pattern_recognition', 'logical_reasoning']
        
        for skill in skill_categories:
            skill_scores = []
            for student_data in student_data_list:
                skill_mastery = student_data.get('skill_mastery', {})
                skill_scores.append(skill_mastery.get(skill, 0.0))
            
            skill_scores = np.array(skill_scores)
            
            # Identify students with gaps in this skill
            gap_threshold = np.percentile(skill_scores, 25)  # Bottom 25%
            students_with_gaps = np.where(skill_scores < gap_threshold)[0]
            
            skill_gaps[skill] = {
                'gap_threshold': gap_threshold,
                'students_with_gaps': len(students_with_gaps),
                'gap_percentage': len(students_with_gaps) / len(skill_scores) * 100,
                'average_gap_severity': np.mean(skill_scores[students_with_gaps]) if len(students_with_gaps) > 0 else 0.0,
                'recommendations': self._get_skill_recommendations(skill, skill_scores)
            }
        
        return skill_gaps
    
    def _get_skill_recommendations(self, skill: str, skill_scores: np.ndarray) -> List[str]:
        """Get recommendations for improving a specific skill."""
        recommendations = {
            'problem_solving': [
                'Practice with step-by-step problem decomposition',
                'Work on pattern recognition exercises',
                'Engage in real-world problem-solving scenarios'
            ],
            'critical_thinking': [
                'Analyze arguments and identify logical fallacies',
                'Practice evaluating evidence and sources',
                'Engage in debate and discussion activities'
            ],
            'analytical_reasoning': [
                'Practice breaking down complex problems',
                'Work on data analysis and interpretation',
                'Engage in logical reasoning exercises'
            ],
            'memory_recall': [
                'Use spaced repetition techniques',
                'Practice active recall methods',
                'Engage in memory palace techniques'
            ],
            'pattern_recognition': [
                'Practice identifying patterns in data',
                'Work on visual pattern recognition exercises',
                'Engage in sequence and series problems'
            ],
            'logical_reasoning': [
                'Practice formal logic exercises',
                'Work on syllogism and deduction problems',
                'Engage in logical puzzle solving'
            ]
        }
        
        return recommendations.get(skill, ['General practice and review'])
    
    def detect_gaps_for_student(self, student_data: Dict) -> Dict[str, Any]:
        """Detect knowledge gaps for a specific student."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        features = self._extract_skill_features(student_data)
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        cluster_label = self.clusterer.predict(features_scaled)[0]
        
        # Detect anomalies
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # Identify specific gaps
        gaps = []
        skill_mastery = student_data.get('skill_mastery', {})
        
        for skill, mastery in skill_mastery.items():
            if mastery < 0.4:  # Gap threshold
                gaps.append({
                    'skill': skill,
                    'mastery_level': mastery,
                    'gap_severity': 'high' if mastery < 0.2 else 'medium',
                    'recommendations': self._get_skill_recommendations(skill, [mastery])
                })
        
        # Overall gap assessment
        overall_gap_score = len(gaps) / len(skill_mastery) if skill_mastery else 0
        
        return {
            'cluster_assignment': int(cluster_label),
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(anomaly_score),
            'identified_gaps': gaps,
            'overall_gap_score': overall_gap_score,
            'gap_severity': 'high' if overall_gap_score > 0.5 else 'medium' if overall_gap_score > 0.2 else 'low',
            'recommendations': self._get_overall_recommendations(gaps, overall_gap_score)
        }
    
    def _get_overall_recommendations(self, gaps: List[Dict], overall_gap_score: float) -> List[str]:
        """Get overall recommendations based on gap analysis."""
        recommendations = []
        
        if overall_gap_score > 0.5:
            recommendations.append("Consider foundational review and remediation")
            recommendations.append("Focus on basic concepts before advanced topics")
        elif overall_gap_score > 0.2:
            recommendations.append("Targeted practice in identified weak areas")
            recommendations.append("Consider additional support or tutoring")
        else:
            recommendations.append("Continue current learning path with minor adjustments")
        
        # Add specific skill recommendations
        for gap in gaps[:3]:  # Top 3 gaps
            recommendations.extend(gap['recommendations'][:2])  # Top 2 recommendations per gap
        
        return recommendations
    
    def get_cluster_insights(self) -> Dict[str, Any]:
        """Get insights about identified clusters."""
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        return {
            'n_clusters': self.n_clusters,
            'clustering_method': self.clustering_method,
            'gap_patterns': self.gap_patterns,
            'skill_gaps': self.skill_gaps
        }
    
    def save_model(self, filepath: str):
        """Save model to file."""
        model_data = {
            'scaler': self.scaler,
            'clusterer': self.clusterer,
            'anomaly_detector': self.anomaly_detector,
            'clustering_method': self.clustering_method,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted,
            'skill_clusters': self.skill_clusters,
            'gap_patterns': self.gap_patterns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.clusterer = model_data['clusterer']
        self.anomaly_detector = model_data['anomaly_detector']
        self.clustering_method = model_data['clustering_method']
        self.n_clusters = model_data['n_clusters']
        self.is_fitted = model_data['is_fitted']
        self.skill_clusters = model_data['skill_clusters']
        self.gap_patterns = model_data['gap_patterns']
