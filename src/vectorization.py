"""
Text vectorization module for cyberbullying detection.
Implements Bag of Words (BoW) and TF-IDF vectorization.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from typing import Tuple, Dict, Any, List, Optional
import pickle
import os


class TextVectorizer:
    """Text vectorization for tweet classification."""
    
    def __init__(
        self,
        vectorizer_type: str = 'tfidf',
        max_features: int = 5000,
        keyword_phrases: Optional[List[str]] = None,
        keyword_feature_weight: float = 3.0,
    ):
        """
        Initialize vectorizer.
        
        Args:
            vectorizer_type: 'bow', 'tfidf', 'tfidf_hybrid', or 'tfidf_hybrid_keywords'
            max_features: Maximum number of features
            keyword_phrases: Optional phrase list for directed keyword features
            keyword_feature_weight: Multiplicative weight for keyword features in hybrid mode
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.keyword_feature_weight = keyword_feature_weight
        self.keyword_phrases = keyword_phrases or [
            'go back',
            'go back your country',
            'go back your country not want you',
            'your country',
            'your country not want you',
            'country',
            'not want you',
            'not welcome here',
            'nobody want you',
            'leave country',
            'do not belong',
            'you do not belong',
            'you should leave',
            'send you back',
            'foreign',
            'immigrant',
            'not welcome',
            'we do not want you',
        ]
        
        self.vectorizer = None
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.keyword_vectorizer = None

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
        elif vectorizer_type == 'tfidf_hybrid':
            self.word_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            self.char_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=max(1000, max_features // 2)
            )
        elif vectorizer_type == 'tfidf_hybrid_keywords':
            self.word_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
            self.char_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=max(1000, max_features // 2)
            )
            keyword_vocab = {phrase: i for i, phrase in enumerate(self.keyword_phrases)}
            # Fixed-vocabulary binary n-grams emphasize directed harassment cues.
            self.keyword_vectorizer = CountVectorizer(
                vocabulary=keyword_vocab,
                ngram_range=(1, 6),
                binary=True,
                lowercase=False,
                token_pattern=r'(?u)\b\w+\b',
            )
        else:
            raise ValueError("vectorizer_type must be 'bow', 'tfidf', 'tfidf_hybrid', or 'tfidf_hybrid_keywords'")
    
    def fit(self, texts: pd.Series) -> 'TextVectorizer':
        """
        Fit vectorizer on training texts.
        
        Args:
            texts: Series of cleaned tweets
        
        Returns:
            Self for chaining
        """
        if self.vectorizer_type == 'tfidf_hybrid':
            self.word_vectorizer.fit(texts)
            self.char_vectorizer.fit(texts)
        elif self.vectorizer_type == 'tfidf_hybrid_keywords':
            self.word_vectorizer.fit(texts)
            self.char_vectorizer.fit(texts)
            self.keyword_vectorizer.fit(texts)
        else:
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
        if self.vectorizer_type == 'tfidf_hybrid':
            # Combine word-level and character-level signals in a single sparse matrix.
            word_matrix = self.word_vectorizer.transform(texts)
            char_matrix = self.char_vectorizer.transform(texts)
            return hstack([word_matrix, char_matrix], format='csr')
        if self.vectorizer_type == 'tfidf_hybrid_keywords':
            word_matrix = self.word_vectorizer.transform(texts)
            char_matrix = self.char_vectorizer.transform(texts)
            keyword_matrix = self.keyword_vectorizer.transform(texts).astype(np.float32) * self.keyword_feature_weight
            return hstack([word_matrix, char_matrix, keyword_matrix], format='csr')
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: pd.Series):
        """
        Fit and transform texts in one step.
        
        Args:
            texts: Series of cleaned tweets
        
        Returns:
            Sparse matrix with vectorized texts
        """
        if self.vectorizer_type == 'tfidf_hybrid':
            word_matrix = self.word_vectorizer.fit_transform(texts)
            char_matrix = self.char_vectorizer.fit_transform(texts)
            return hstack([word_matrix, char_matrix], format='csr')
        if self.vectorizer_type == 'tfidf_hybrid_keywords':
            word_matrix = self.word_vectorizer.fit_transform(texts)
            char_matrix = self.char_vectorizer.fit_transform(texts)
            keyword_matrix = self.keyword_vectorizer.fit_transform(texts).astype(np.float32) * self.keyword_feature_weight
            return hstack([word_matrix, char_matrix, keyword_matrix], format='csr')
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self):
        """Get feature names (vocabulary)."""
        if self.vectorizer_type == 'tfidf_hybrid':
            word_features = [f"w:{f}" for f in self.word_vectorizer.get_feature_names_out()]
            char_features = [f"c:{f}" for f in self.char_vectorizer.get_feature_names_out()]
            return np.array(word_features + char_features)
        if self.vectorizer_type == 'tfidf_hybrid_keywords':
            word_features = [f"w:{f}" for f in self.word_vectorizer.get_feature_names_out()]
            char_features = [f"c:{f}" for f in self.char_vectorizer.get_feature_names_out()]
            keyword_features = [f"k:{f}" for f in self.keyword_phrases]
            return np.array(word_features + char_features + keyword_features)
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath: str):
        """Save vectorizer to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            if self.vectorizer_type in ('tfidf_hybrid', 'tfidf_hybrid_keywords'):
                payload = {
                    'vectorizer_type': self.vectorizer_type,
                    'max_features': self.max_features,
                    'keyword_feature_weight': self.keyword_feature_weight,
                    'word_vectorizer': self.word_vectorizer,
                    'char_vectorizer': self.char_vectorizer,
                    'keyword_vectorizer': self.keyword_vectorizer,
                    'keyword_phrases': self.keyword_phrases,
                }
                pickle.dump(payload, f)
            else:
                pickle.dump(self.vectorizer, f)
    
    def load(self, filepath: str) -> 'TextVectorizer':
        """Load vectorizer from file."""
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict) and loaded.get('vectorizer_type') in ('tfidf_hybrid', 'tfidf_hybrid_keywords'):
            self.vectorizer_type = loaded.get('vectorizer_type', 'tfidf_hybrid')
            self.max_features = loaded.get('max_features', self.max_features)
            self.keyword_feature_weight = loaded.get('keyword_feature_weight', self.keyword_feature_weight)
            self.vectorizer = None
            self.word_vectorizer = loaded['word_vectorizer']
            self.char_vectorizer = loaded['char_vectorizer']
            self.keyword_vectorizer = loaded.get('keyword_vectorizer')
            self.keyword_phrases = loaded.get('keyword_phrases', self.keyword_phrases)
        else:
            self.vectorizer = loaded
            self.word_vectorizer = None
            self.char_vectorizer = None
            self.keyword_vectorizer = None
        return self


def vectorize_texts(X_train: pd.Series, X_test: pd.Series, 
                    vectorizer_type: str = 'tfidf', 
                    max_features: int = 5000,
                    keyword_phrases: Optional[List[str]] = None,
                    keyword_feature_weight: float = 3.0) -> Tuple[Any, Any, TextVectorizer]:
    """
    Vectorize training and test texts.
    
    Args:
        X_train: Training tweets
        X_test: Test tweets
        vectorizer_type: Type of vectorization
        max_features: Maximum features
        keyword_phrases: Optional phrase list for directed keyword features
        keyword_feature_weight: Multiplicative weight for keyword features
    
    Returns:
        Tuple of (X_train_vec, X_test_vec, vectorizer)
    """
    vectorizer = TextVectorizer(vectorizer_type, max_features, keyword_phrases, keyword_feature_weight)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, vectorizer
