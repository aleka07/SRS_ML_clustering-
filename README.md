# ArXiv Scientific Paper Clustering Library

This Python library provides tools to automatically cluster scientific articles from the arXiv dataset based on their summaries (abstracts) using machine learning techniques. It follows an object-oriented design for modularity and extensibility.

## Features

*   **Data Loading:** Loads article metadata from the arXiv dataset CSV file.
*   **Text Preprocessing:** Cleans and prepares article summaries, including:
    *   Lowercasing
    *   Punctuation removal
    *   Stopword removal (common English words)
    *   Lemmatization (reducing words to their base form)
*   **Text Vectorization:** Converts processed text into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) method.
*   **Clustering Algorithms:** Implements common clustering algorithms:
    *   K-Means
    *   Agglomerative Hierarchical Clustering
*   **Evaluation:** Assesses the quality of the resulting clusters using standard metrics:
    *   Silhouette Score
    *   Davies-Bouldin Index
*   **Visualization:** Creates 2D scatter plots of the clusters for visual inspection, using dimensionality reduction techniques:
    *   PCA (Principal Component Analysis)
    *   t-SNE (t-Distributed Stochastic Neighbor Embedding)
    *   TruncatedSVD (Singular Value Decomposition for sparse data)
*   **Command-Line Interface:** Allows running the pipeline with configurable parameters.

## Project Structure

```
arxiv_clustering_lib/
├── data_loader/
│   ├── __init__.py
│   └── loader.py            # Module for loading data
├── preprocessing/
│   ├── __init__.py
│   ├── text_cleaner.py      # Module for text cleaning
│   └── vectorizer.py        # Module for text vectorization (TF-IDF)
├── clustering/
│   ├── __init__.py
│   ├── kmeans.py            # Module for K-Means clustering
│   └── hierarchical.py      # Module for Hierarchical clustering
├── evaluation/
│   ├── __init__.py
│   └── metrics.py           # Module for calculating clustering metrics
├── visualization/
│   ├── __init__.py
│   └── plot_clusters.py     # Module for visualizing clusters
├── main.py                  # Main script to execute the pipeline
├── requirements.txt         # List of required Python packages
└── README.md                # This documentation file
```
      
## Prerequisites

1.  **Python:** Version 3.8 or higher is recommended.
2.  **Dependencies:** Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  **NLTK Data:** Download necessary data files for text processing (run this in a Python interpreter once):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    ```

## Dataset

This project uses the arXiv Scientific Research Papers Dataset available on Kaggle:
[https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset/data](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset/data)

*   Download the `arxivData.csv` file from the link above.
*   Place the `arxivData.csv` file inside the main `arxiv_clustering_lib` directory, or provide the correct path using the `-f` command-line option when running `main.py`.

## Usage

Execute the clustering pipeline by running the `main.py` script from your terminal within the `arxiv_clustering_lib` directory.

**Basic Command:**

```bash
python main.py [OPTIONS]
```

## Command-line Options

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| --file | -f | Path to the input CSV dataset. | arxivData.csv |
| --sample | -s | Number of articles to sample from the dataset. Use 0 or a negative number for the full dataset. Processing fewer samples is faster. | 2000 |
| --clusters | -k | The desired number of clusters (K). | 10 |
| --algo | -a | The clustering algorithm to use. Choices: kmeans, hierarchical. | kmeans |
| --vis | -v | The dimensionality reduction method for visualization. Choices: pca, tsne, svd. | tsne |
| --saveplot |  | File path to save the generated cluster plot (e.g., cluster_plot.png). If omitted, the plot is displayed interactively. | None |
| --savecsv |  | File path to save the results (original data + cluster labels) as a CSV (e.g., clustered_data.csv). If omitted, results are not saved. | None |

## Examples

* **Run with defaults:** Use K-Means (k=10) on 2000 samples, visualize with t-SNE, display plot.
  ```bash
  python main.py
  ```

* **Run K-Means with specific parameters:** Use K-Means (k=15) on 5000 samples, visualize with t-SNE, save plot and output CSV.
  ```bash
  python main.py -s 5000 -k 15 -a kmeans -v tsne --saveplot kmeans_tsne_k15.png --savecsv results_k15.csv
  ```

* **Run Hierarchical Clustering:** Use Hierarchical clustering (k=8) on 1000 samples, visualize with PCA, save plot.
  ```bash
  python main.py -s 1000 -k 8 -a hierarchical -v pca --saveplot hier_pca_k8.png
  ```

* **Run on full dataset (Caution: slow and memory-intensive):** Use K-Means (k=20) on all data, visualize with SVD (faster for large sparse data), save plot.
  ```bash
  python main.py -s 0 -k 20 -a kmeans -v svd --saveplot full_kmeans_svd_k20.png
  ```

## Customization and Extension

* **Parameters:** Easily adjust pipeline parameters (like tfidf_max_features or hierarchical_linkage) by modifying the pipeline_config dictionary in main.py or adding more command-line arguments.

* **New Vectorizers:** Add support for other text vectorization methods (e.g., Word2Vec, Doc2Vec, BERT embeddings) by creating new classes or functions within preprocessing/vectorizer.py and updating main.py.

* **New Clustering Algorithms:** Implement additional clustering algorithms (e.g., DBSCAN, Spectral Clustering) by adding modules to the clustering/ directory and integrating them into main.py.

* **Configuration Files:** Modify main.py to load configurations from external files (like YAML or JSON) for more complex setups.

## Important Notes

* **Memory Usage:** Processing large datasets, especially with TF-IDF, t-SNE, and Hierarchical clustering ('ward' linkage), can consume significant RAM. Running on a smaller sample (-s option) is recommended for initial exploration or on machines with limited memory.

* **Performance:**
  * t-SNE provides often insightful visualizations but is computationally expensive. PCA or SVD are faster alternatives.
  * Hierarchical clustering can be slow, especially on large datasets.
  * TF-IDF calculation time depends on the vocabulary size (max_features) and dataset size.

* **Clustering Quality:** The effectiveness of the clustering depends heavily on the preprocessing steps, vectorization method, chosen algorithm, the number of clusters (K), and the inherent structure of the data. Experimentation with parameters is usually necessary to achieve meaningful results. The provided evaluation metrics help quantify the quality of the chosen configuration.