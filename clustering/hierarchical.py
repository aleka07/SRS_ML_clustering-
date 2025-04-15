# clustering/hierarchical.py
from sklearn.cluster import AgglomerativeClustering
import logging
import numpy as np
from scipy.sparse import csr_matrix

class HierarchicalClusterer:
    """
    Performs Agglomerative Hierarchical Clustering.
    """
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward'):
        """
        Args:
            n_clusters (int): The desired number of clusters.
            linkage (str): Which linkage criterion to use ('ward', 'complete', 'average', 'single').
                           'ward' is often a good default but requires Euclidean distance (dense data).
        """
        if n_clusters <= 1:
             raise ValueError("Number of clusters (n_clusters) must be greater than 1.")
        self.n_clusters = n_clusters
        self.linkage = linkage
        # Note: AgglomerativeClustering doesn't have random_state
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             linkage=self.linkage,
                                             # metric='euclidean' is default for ward
                                             # metric='cosine' could be used with other linkages
                                             )
        self.labels_ = None
        logging.info(f"HierarchicalClusterer initialized with n_clusters={n_clusters}, linkage='{linkage}'")

    def fit_predict(self, data: csr_matrix | np.ndarray) -> np.ndarray | None:
        """
        Fits the Hierarchical Clustering model and returns cluster labels.

        Args:
            data (csr_matrix | np.ndarray): The vectorized data.
                IMPORTANT: If using 'ward' linkage, data MUST be dense (or converted).
                           Other linkages might support sparse data with appropriate metrics,
                           but performance can vary.

        Returns:
            np.ndarray | None: Array of cluster labels for each data point, or None if failed.
        """
        logging.info(f"Starting Hierarchical clustering with {self.n_clusters} clusters (linkage='{self.linkage}')...")

        # --- Data Density Check ---
        data_dense = data
        is_sparse = isinstance(data, csr_matrix)
        if self.linkage == 'ward' and is_sparse:
            logging.warning("Linkage 'ward' requires dense data. Converting sparse matrix to dense array. This may consume significant memory.")
            try:
                data_dense = data.toarray()
            except MemoryError:
                logging.error("MemoryError: Cannot convert sparse matrix to dense for 'ward' linkage. Try reducing data size or using a different linkage method/metric.")
                return None
            except Exception as e:
                 logging.error(f"Error converting sparse matrix to dense: {e}")
                 return None
        elif is_sparse and self.linkage != 'ward':
             # Check if the combination is supported or if dense conversion is still better
             # For simplicity, let's try running directly, but be aware it might be slow or ineffective
             logging.warning(f"Using linkage '{self.linkage}' with sparse data. Performance/results may vary. Consider cosine metric if applicable.")
             # AgglomerativeClustering doesn't directly support sparse input well for most linkage/metric combos
             # Often better to reduce dimensionality first (e.g., PCA/TruncatedSVD) then cluster the dense reduced data.
             # For this example, we'll attempt direct clustering but recommend dense conversion or dimensionality reduction.
             logging.warning("Attempting direct clustering on sparse data. It's often better to convert to dense or use dimensionality reduction first for Hierarchical Clustering.")


        try:
            self.labels_ = self.model.fit_predict(data_dense) # Use potentially converted data
            logging.info(f"Hierarchical clustering completed. Found {len(np.unique(self.labels_))} clusters.")
            return self.labels_
        except MemoryError:
             logging.error(f"MemoryError during Hierarchical clustering. Input data shape: {data_dense.shape}. Consider reducing data size or dimensionality.")
             return None
        except Exception as e:
            logging.error(f"Error during Hierarchical clustering: {e}")
            return None

    def get_labels(self) -> np.ndarray | None:
        """Returns the computed cluster labels."""
        return self.labels_