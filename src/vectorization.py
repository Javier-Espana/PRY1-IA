"""
Text vectorization module for cyberbullying detection.
Implements Bag of Words (BoW) and TF-IDF vectorization.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Tuple, Dict, Any
import pickle
import os


class TextVectorizer:
    """Text vectorization for tweet classification."""
    
    def __init__(self, vectorizer_type: str = 'tfidf', max_features: int = 5000):
        """
        Initialize vectorizer.
        
        Args:
            vectorizer_type: 'bow' for Bag of Words, 'tfidf' for TF-IDF
            max_features: Maximum number of features
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        
        if vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        else:
            raise ValueError("vectorizer_type must be 'bow' or 'tfidf'")
    
    def fit(self, texts: pd.Series) -> 'TextVectorizer':
        """
        Fit vectorizer on training texts.
        
        Args:
            texts: Series of cleaned tweets
        
        Returns:
            Self for chaining
        """
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts: pd.Series):
        """
        Transform texts to vector space.
        
        Args:
            texts: Series of cleaned tweets
        
        Returns:
            Sparse matrix with vectorized texts
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: pd.Series):
        """
        Fit and transform texts in one step.
        
        Args:
            texts: Series of cleaned tweets
        
        Returns:
            Sparse matrix with vectorized texts
        """
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self):
        """Get feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath: str):
        """Save vectorizer to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, filepath: str) -> 'TextVectorizer':
        """Load vectorizer from file."""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        return self


def vectorize_texts(X_train: pd.Series, X_test: pd.Series, 
                    vectorizer_type: str = 'tfidf', 
                    max_features: int = 5000) -> Tuple[Any, Any, TextVectorizer]:
    """
    Vectorize training and test texts.
    
    Args:
        X_train: Training tweets
        X_test: Test tweets
        vectorizer_type: Type of vectorization
        max_features: Maximum features
    
    Returns:
        Tuple of (X_train_vec, X_test_vec, vectorizer)
    """
    vectorizer = TextVectorizer(vectorizer_type, max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, vectorizer
