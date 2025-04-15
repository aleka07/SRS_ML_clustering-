# preprocessing/text_cleaner.py
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

# Ensure NLTK data is available (run nltk.download(...) if needed)
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    logging.error("NLTK data not found. Please run nltk.download('punkt'), nltk.download('stopwords'), nltk.download('wordnet'), nltk.download('omw-1.4')")
    raise

class TextCleaner:
    """
    Cleans and preprocesses text data, specifically summaries.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add custom stopwords if needed
        # self.stop_words.update(['figure', 'table', 'paper', 'study', 'result', 'method', 'approach'])

    def _clean_individual_text(self, text: str) -> str:
        """Applies cleaning steps to a single text string."""
        if not isinstance(text, str):
            return "" # Handle potential non-string inputs (like NaN)

        # 1. Lowercase
        text = text.lower()
        # 2. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 3. Remove numbers (optional, can be kept if relevant)
        text = re.sub(r'\d+', '', text)
        # 4. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize_and_process(self, text: str) -> str:
        """Tokenizes, removes stopwords, and lemmatizes."""
        tokens = word_tokenize(text)
        # Remove stopwords and short tokens
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens) # Return processed text as a single string

    def preprocess_series(self, text_series: pd.Series) -> pd.Series:
        """
        Applies the full preprocessing pipeline to a pandas Series of text.

        Args:
            text_series (pd.Series): The Series containing text data (e.g., summaries).

        Returns:
            pd.Series: The Series with preprocessed text.
        """
        logging.info("Starting text preprocessing...")
        # Handle missing values
        cleaned_series = text_series.fillna('').astype(str)
        # Apply basic cleaning
        cleaned_series = cleaned_series.apply(self._clean_individual_text)
        # Apply tokenization, stopword removal, lemmatization
        processed_series = cleaned_series.apply(self._tokenize_and_process)
        logging.info("Text preprocessing completed.")
        return processed_series