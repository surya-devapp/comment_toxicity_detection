from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from textblob import TextBlob
import re

class SentimentExtractor(BaseEstimator, TransformerMixin):
    """Extracts sentiment polarity from text."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Expecting X to be a pandas Series or list of strings
        sentiments = []
        for text in X:
            try:
                blob = TextBlob(str(text))
                sentiments.append(blob.sentiment.polarity)
            except:
                sentiments.append(0.0)
        return np.array(sentiments).reshape(-1, 1)

class TargetExtractor(BaseEstimator, TransformerMixin):
    """Detects presence of 2nd person pronouns indicating a personal target."""
    def __init__(self):
        self.pronouns = {"you", "your", "you're", "u", "ur", "yourself"}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_lower = str(text).lower()
            words = set(re.findall(r'\b\w+\b', text_lower))
            # Check overlap
            if words & self.pronouns:
                features.append(1)
            else:
                features.append(0)
        return np.array(features).reshape(-1, 1)

class LengthExtractor(BaseEstimator, TransformerMixin):
    """Extracts length of text relative to a max length."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        lengths = [len(str(t).split()) for t in X]
        return np.array(lengths).reshape(-1, 1)
