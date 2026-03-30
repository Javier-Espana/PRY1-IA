"""
Text preprocessing module for cyberbullying detection.
Handles cleaning, normalization, tokenization, and lemmatization.
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # Required by newer NLTK versions for sentence tokenization resources.
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class TextPreprocessor:
    """Text preprocessing for cyberbullying tweets."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text."""
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text: str) -> str:
        """Remove hashtags from text."""
        return re.sub(r'#\w+', '', text)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def remove_special_chars(self, text: str) -> str:
        """Remove special characters, keep alphanumeric and spaces."""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def normalize_text(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common English stopwords."""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduce words to their base form (lemmatization)."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        # preserve_line avoids sentence tokenization dependency on punkt resources.
        return word_tokenize(text, preserve_line=True)
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps."""
        # Remove URLs
        text = self.remove_urls(text)
        # Remove mentions
        text = self.remove_mentions(text)
        # Remove hashtags
        text = self.remove_hashtags(text)
        # Remove emojis
        text = self.remove_emojis(text)
        # Remove special characters
        text = self.remove_special_chars(text)
        # Normalize to lowercase
        text = self.normalize_text(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_text(self, text: str) -> str:
        """Complete text processing: clean, tokenize, remove stopwords, lemmatize."""
        # Clean text
        text = self.clean_text(text)
        # Tokenize
        tokens = self.tokenize(text)
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        # Lemmatize
        tokens = self.lemmatize(tokens)
        # Join back to string
        return ' '.join(tokens)
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'tweet_text') -> pd.DataFrame:
        """Process all tweets in a dataframe."""
        df_copy = df.copy()
        df_copy[f'{text_column}_cleaned'] = df_copy[text_column].apply(self.process_text)
        return df_copy


def preprocess_data(df: pd.DataFrame, text_column: str = 'tweet_text') -> pd.DataFrame:
    """
    Convenience function to preprocess data.
    
    Args:
        df: DataFrame with tweets
        text_column: Name of the column containing tweets
    
    Returns:
        Processed dataframe with cleaned tweets
    """
    preprocessor = TextPreprocessor()
    return preprocessor.process_dataframe(df, text_column)
