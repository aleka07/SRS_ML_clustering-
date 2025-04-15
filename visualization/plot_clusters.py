# visualization/plot_clusters.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import logging
import numpy as np
from scipy.sparse import csr_matrix

class ClusterVisualizer:
    """
    Visualizes clustering results using dimensionality reduction.
    """
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None

    def _reduce_dimensions(self, data: csr_matrix | np.ndarray, method: str = 'pca') -> np.ndarray | None:
        """
        Reduces the dimensionality of the data.

        Args:
            data (csr_matrix | np.ndarray): High-dimensional data (sparse or dense).
            method (str): Reduction method ('pca', 'tsne', 'svd'). 'pca' and 'svd' handle sparse.
                          'tsne' typically requires dense data.

        Returns:
            np.ndarray | None: Lower-dimensional representation of the data, or None on failure.
        """
        logging.info(f"Reducing dimensionality using {method.upper()} to {self.n_components} components...")
        is_sparse = isinstance(data, csr_matrix)

        try:
            if method == 'pca':
                # PCA in scikit-learn >= 0.18 handles sparse input (implicitly uses TruncatedSVD)
                self.reducer = PCA(n_components=self.n_components, random_state=self.random_state)
                reduced_data = self.reducer.fit_transform(data if not is_sparse else data.toarray()) # PCA expects dense
                # Note: for very high dim sparse, TruncatedSVD is better choice
            elif method == 'svd':
                 # TruncatedSVD works directly on sparse matrices
                 self.reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
                 reduced_data = self.reducer.fit_transform(data)
            elif method == 'tsne':
                # t-SNE generally requires dense data and can be memory/CPU intensive
                data_dense = data
                if is_sparse:
                    logging.warning("t-SNE requires dense data. Converting sparse matrix. This may consume significant memory/time.")
                    try:
                        # Maybe reduce with SVD first if dimensions are huge before t-SNE
                        if data.shape[1] > 100:
                             logging.info("Applying TruncatedSVD before t-SNE for efficiency.")
                             svd = TruncatedSVD(n_components=50, random_state=self.random_state) # Reduce to intermediate dim
                             data_dense = svd.fit_transform(data)
                        else:
                             data_dense = data.toarray()

                    except MemoryError:
                         logging.error("MemoryError: Cannot convert sparse matrix to dense for t-SNE. Try 'svd' or 'pca' instead, or reduce data size.")
                         return None
                    except Exception as e:
                         logging.error(f"Error preparing data for t-SNE: {e}")
                         return None

                self.reducer = TSNE(n_components=self.n_components, random_state=self.random_state,
                                    perplexity=min(30, data_dense.shape[0] - 1), # Adjust perplexity if needed
                                    n_iter=300, # Faster for visualization
                                    init='pca', # Use PCA init
                                    learning_rate='auto'
                                    )
                reduced_data = self.reducer.fit_transform(data_dense)
            else:
                logging.error(f"Unsupported dimensionality reduction method: {method}")
                return None

            logging.info(f"Dimensionality reduction complete. Output shape: {reduced_data.shape}")
            return reduced_data

        except Exception as e:
            logging.error(f"Error during dimensionality reduction with {method.upper()}: {e}")
            return None

    def plot(self, data: csr_matrix | np.ndarray, labels: np.ndarray,
             method: str = 'pca', title: str = 'Cluster Visualization',
             save_path: str | None = None):
        """
        Reduces dimensions and creates a scatter plot of the clusters.

        Args:
            data (csr_matrix | np.ndarray): Original high-dimensional vectorized data.
            labels (np.ndarray): Cluster labels for each data point.
            method (str): Dimensionality reduction method ('pca', 'tsne', 'svd').
            title (str): Title for the plot.
            save_path (str | None): If provided, saves the plot to this file path instead of showing it.
        """
        reduced_data = self._reduce_dimensions(data, method=method)

        if reduced_data is None:
            logging.error("Cannot plot clusters due to dimensionality reduction failure.")
            return

        if reduced_data.shape[0] != len(labels):
             logging.error(f"Mismatch between reduced data points ({reduced_data.shape[0]}) and labels ({len(labels)}). Cannot plot.")
             return

        plt.figure(figsize=(12, 8))
        n_clusters = len(np.unique(labels))
        sns.scatterplot(
            x=reduced_data[:, 0],
            y=reduced_data[:, 1],
            hue=labels,
            palette=sns.color_palette("hsv", n_clusters), # Use a palette suitable for categories
            legend="full",
            alpha=0.7
        )
        plt.title(f'{title} ({method.upper()} Projection)')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Cluster plot saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save plot to {save_path}: {e}")
                plt.show() # Show plot if saving failed
        else:
            plt.show()
        plt.close() # Close the plot figure