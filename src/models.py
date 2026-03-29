"""
Machine learning models for cyberbullying detection.
Implements and compares multiple algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Any, Tuple
import pickle
import os


class CyberbullyingClassifier:
    """Wrapper for cyberbullying classification models."""
    
    def __init__(self, model_name: str, model=None, seed: int = 42):
        """
        Initialize classifier.
        
        Args:
            model_name: Name of the model
            model: sklearn model instance
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.model = model
        self.seed = seed
        self.metrics = {}
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return self.model.decision_function(X)
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        self.metrics = metrics
        self.y_test = y_test
        self.y_pred = y_pred
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix."""
        if hasattr(self, 'y_pred'):
            return confusion_matrix(self.y_test, self.y_pred)
        return None
    
    def get_classification_report(self):
        """Get detailed classification report."""
        if hasattr(self, 'y_pred'):
            return classification_report(self.y_test, self.y_pred)
        return None
    
    def save(self, filepath: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        return self


def create_models(seed: int = 42) -> Dict[str, CyberbullyingClassifier]:
    """
    Create all models to be compared.
    
    Args:
        seed: Random seed
    
    Returns:
        Dictionary with model instances
    """
    models = {}
    
    # Naive Bayes - probabilistic classifier, good for text
    models['Naive Bayes'] = CyberbullyingClassifier(
        'Naive Bayes',
        MultinomialNB(alpha=0.01),
        seed
    )
    
    # Logistic Regression - linear model for classification
    models['Logistic Regression'] = CyberbullyingClassifier(
        'Logistic Regression',
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight='balanced'
        ),
        seed
    )
    
    # SVM - powerful for text classification
    models['SVM'] = CyberbullyingClassifier(
        'Support Vector Machine (SVM)',
        LinearSVC(
            max_iter=2000,
            random_state=seed,
            class_weight='balanced',
            dual=False
        ),
        seed
    )
    
    # Gradient Boosting - ensemble method with strong performance
    models['Gradient Boosting'] = CyberbullyingClassifier(
        'Gradient Boosting',
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=seed
        ),
        seed
    )
    
    # Neural Network - deep learning approach
    models['Neural Network'] = CyberbullyingClassifier(
        'Neural Network (MLP)',
        MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            batch_size=32,
            learning_rate_init=0.001,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1
        ),
        seed
    )
    
    return models


def train_and_evaluate_models(models: Dict[str, CyberbullyingClassifier],
                              X_train, y_train,
                              X_test, y_test) -> pd.DataFrame:
    """
    Train and evaluate all models.
    
    Args:
        models: Dictionary of models
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, classifier in models.items():
        print(f"\nTraining {name}...")
        classifier.train(X_train, y_train)
        
        print(f"Evaluating {name}...")
        metrics = classifier.evaluate(X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'accuracy', 'precision', 'recall', 'f1']]
    
    return results_df
