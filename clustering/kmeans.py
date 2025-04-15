# clustering/kmeans.py
from sklearn.cluster import KMeans
import logging
import numpy as np
from scipy.sparse import csr_matrix

class KMeansClusterer:
    """
    Performs K-Means clustering.
    """
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Args:
            n_clusters (int): The desired number of clusters.
            random_state (int): Seed for reproducibility.
        """
        if n_clusters <= 1:
             raise ValueError("Number of clusters (n_clusters) must be greater than 1.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters,
                            random_state=self.random_state,
                            n_init=10) # n_init > 1 improves stability
        self.labels_ = None
        logging.info(f"KMeansClusterer initialized with n_clusters={n_clusters}")

    def fit_predict(self, data: csr_matrix | np.ndarray) -> np.ndarray | None:
        """
        Fits the K-Means model to the data and returns cluster labels.

        Args:
            data (csr_matrix | np.ndarray): The vectorized data (sparse or dense).

        Returns:
            np.ndarray | None: Array of cluster labels for each data point, or None if failed.
        """
        logging.info(f"Starting K-Means clustering with {self.n_clusters} clusters...")
        try:
            # KMeans in scikit-learn handles sparse matrices efficiently
            self.labels_ = self.model.fit_predict(data)
            logging.info(f"K-Means clustering completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
        except Exception as e:
            logging.error(f"Error during K-Means clustering: {e}")
            return None

    def get_labels(self) -> np.ndarray | None:
        """Returns the computed cluster labels."""
        return self.labels_