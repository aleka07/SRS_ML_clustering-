# evaluation/metrics.py
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging
import numpy as np
from scipy.sparse import csr_matrix

class ClusteringEvaluator:
    """
    Calculates internal clustering evaluation metrics.
    """
    def __init__(self):
        pass

    def calculate_metrics(self, data: csr_matrix | np.ndarray, labels: np.ndarray) -> dict:
        """
        Calculates Silhouette Score and Davies-Bouldin Index.

        Args:
            data (csr_matrix | np.ndarray): The vectorized data used for clustering.
                                            Sparse matrix is acceptable.
            labels (np.ndarray): The cluster labels assigned to each data point.

        Returns:
            dict: A dictionary containing the calculated metric scores.
                  Returns empty dict if calculation fails or is not possible.
        """
        results = {}
        n_labels = len(np.unique(labels))

        if n_labels <= 1 or n_labels >= data.shape[0]:
            logging.warning(f"Cannot calculate metrics with {n_labels} cluster(s) for {data.shape[0]} samples.")
            return results

        logging.info("Calculating clustering evaluation metrics...")
        try:
            # Silhouette Score: Higher is better (-1 to 1)
            # Use a sample if the dataset is too large, as silhouette can be slow
            sample_size = min(5000, data.shape[0]) # Limit sample size for metric calculation speed
            if data.shape[0] > sample_size:
                logging.info(f"Calculating Silhouette score on a sample of size {sample_size}")
                # Need to sample both data and labels consistently
                indices = np.random.choice(data.shape[0], sample_size, replace=False)
                data_sample = data[indices]
                labels_sample = labels[indices]
                # Ensure the sample still has enough clusters
                if len(np.unique(labels_sample)) > 1:
                     silhouette = silhouette_score(data_sample, labels_sample)
                     results['silhouette_score'] = silhouette
                     logging.info(f"Silhouette Score (sample): {silhouette:.4f}")
                else:
                     logging.warning("Sample for Silhouette score resulted in <= 1 cluster. Skipping metric.")
            else:
                 silhouette = silhouette_score(data, labels)
                 results['silhouette_score'] = silhouette
                 logging.info(f"Silhouette Score: {silhouette:.4f}")

        except Exception as e:
            logging.error(f"Error calculating Silhouette Score: {e}")

        try:
            # Davies-Bouldin Index: Lower is better (>= 0)
            # Requires dense data if using default Euclidean distance
            data_dense = data
            if isinstance(data, csr_matrix):
                logging.warning("Davies-Bouldin Index requires dense data. Converting sparse matrix. This may consume memory.")
                try:
                    data_dense = data.toarray()
                except MemoryError:
                    logging.error("MemoryError converting to dense for Davies-Bouldin. Skipping metric.")
                    data_dense = None # Flag that conversion failed
                except Exception as e:
                    logging.error(f"Error converting to dense for Davies-Bouldin: {e}")
                    data_dense = None

            if data_dense is not None:
                 db_index = davies_bouldin_score(data_dense, labels)
                 results['davies_bouldin_index'] = db_index
                 logging.info(f"Davies-Bouldin Index: {db_index:.4f}")

        except Exception as e:
            logging.error(f"Error calculating Davies-Bouldin Index: {e}")

        return results