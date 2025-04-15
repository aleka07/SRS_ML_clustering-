# preprocessing/vectorizer.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec # Example for future Word2Vec addition
import logging
from scipy.sparse import csr_matrix

class Vectorizer:
    """
    Transforms text data into numerical vectors.
    Currently supports TF-IDF.
    """
    def __init__(self, method: str = 'tfidf', max_features: int | None = 5000):
        """
        Args:
            method (str): The vectorization method ('tfidf', potentially 'word2vec' later).
            max_features (int | None): Maximum number of features for TF-IDF.
        """
        if method not in ['tfidf']: # Add 'word2vec' when implemented
             raise ValueError("Unsupported vectorization method. Choose 'tfidf'.")
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.feature_names = None
        logging.info(f"Vectorizer initialized with method: {self.method}, max_features: {self.max_features}")

    def fit_transform(self, texts: pd.Series) -> csr_matrix | None :
        """
        Fits the vectorizer on the text data and transforms it.

        Args:
            texts (pd.Series): A Series of preprocessed text documents.

        Returns:
            scipy.sparse.csr_matrix | None: The document-term matrix, or None if failed.
            Returns None for Word2Vec for now (as it requires different handling).
        """
        logging.info(f"Starting vectorization using {self.method}...")
        if self.method == 'tfidf':
            try:
                self.vectorizer = TfidfVectorizer(max_features=self.max_features,
                                                  stop_words=None) # Stop words already removed
                vectorized_data = self.vectorizer.fit_transform(texts)
                self.feature_names = self.vectorizer.get_feature_names_out()
                logging.info(f"TF-IDF Vectorization complete. Shape: {vectorized_data.shape}")
                return vectorized_data
            except Exception as e:
                 logging.error(f"Error during TF-IDF vectorization: {e}")
                 return None

        # Placeholder for Word2Vec (requires more complex implementation)
        # elif self.method == 'word2vec':
        #     logging.warning("Word2Vec vectorization is not fully implemented yet.")
        #     # Example: Train or load model, then average vectors per document
        #     # sentences = [doc.split() for doc in texts]
        #     # model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)
        #     # self.vectorizer = model # Store the model
        #     # Implement function to get document vectors by averaging word vectors
        #     # return document_vectors
        #     return None

        return None # Should not be reached if method validation works

    def get_feature_names(self) -> list[str] | None:
        """Returns the feature names (vocabulary) learned by the vectorizer."""
        if self.feature_names is not None:
            return list(self.feature_names)
        else:
            logging.warning("Vectorizer has not been fitted yet or does not provide feature names.")
            return None