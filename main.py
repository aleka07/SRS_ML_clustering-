# main.py
import pandas as pd
import logging
import time
import argparse # For command-line arguments

# Import library modules
from data_loader.loader import DataLoader
from preprocessing.text_cleaner import TextCleaner
from preprocessing.vectorizer import Vectorizer
from clustering.kmeans import KMeansClusterer
from clustering.hierarchical import HierarchicalClusterer
from evaluation.metrics import ClusteringEvaluator
from visualization.plot_clusters import ClusterVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline(config):
    """Executes the full clustering pipeline based on the config."""
    start_time = time.time()
    logging.info("Starting ArXiv Clustering Pipeline...")

    # 1. Load Data
    loader = DataLoader()
    df_raw = loader.load_data(config['data_path'])
    if df_raw is None:
        logging.error("Pipeline halted: Data loading failed.")
        return

    # Handle missing summaries crucial for clustering
    df_raw.dropna(subset=['summary'], inplace=True)
    df_raw.reset_index(drop=True, inplace=True)
    if df_raw.empty:
         logging.error("Pipeline halted: No valid data after dropping rows with missing summaries.")
         return

    logging.info(f"Dataframe shape after dropping NaNs in summary: {df_raw.shape}")

    # Optional: Sample data
    df = loader.get_sample(df_raw, n_samples=config.get('sample_size'), random_state=config.get('random_state', 42))
    if df.empty:
         logging.error("Pipeline halted: Sampled dataframe is empty.")
         return
    logging.info(f"Using DataFrame subset of shape: {df.shape}")


    # 2. Preprocess Text
    cleaner = TextCleaner()
    # Make sure to use the correct column name ('summary')
    df['processed_summary'] = cleaner.preprocess_series(df['summary'])

    # Check for empty summaries after processing
    empty_summaries = df[df['processed_summary'].str.strip() == ''].index
    if not empty_summaries.empty:
        logging.warning(f"Found {len(empty_summaries)} entries with empty summaries after preprocessing. Removing them.")
        df = df.drop(empty_summaries).reset_index(drop=True)
        if df.empty:
            logging.error("Pipeline halted: All summaries became empty after preprocessing.")
            return


    # 3. Vectorize Text
    vectorizer = Vectorizer(method=config.get('vectorizer_method', 'tfidf'),
                            max_features=config.get('tfidf_max_features', 5000))
    vectorized_data = vectorizer.fit_transform(df['processed_summary'])
    if vectorized_data is None:
        logging.error("Pipeline halted: Vectorization failed.")
        return


    # 4. Cluster Data
    n_clusters = config.get('n_clusters', 5)
    cluster_algo = config.get('clustering_algorithm', 'kmeans')
    labels = None

    if cluster_algo == 'kmeans':
        clusterer = KMeansClusterer(n_clusters=n_clusters, random_state=config.get('random_state', 42))
        labels = clusterer.fit_predict(vectorized_data)
    elif cluster_algo == 'hierarchical':
        clusterer = HierarchicalClusterer(n_clusters=n_clusters, linkage=config.get('hierarchical_linkage', 'ward'))
        # Hierarchical might need dense data depending on linkage, handled inside its fit_predict
        labels = clusterer.fit_predict(vectorized_data)
    else:
        logging.error(f"Unsupported clustering algorithm: {cluster_algo}. Choose 'kmeans' or 'hierarchical'.")
        return

    if labels is None:
        logging.error("Pipeline halted: Clustering failed.")
        return

    df['cluster_label'] = labels


    # 5. Evaluate Clustering
    evaluator = ClusteringEvaluator()
    # Pass appropriate data form (sparse okay for silhouette, need dense for DB)
    metrics = evaluator.calculate_metrics(vectorized_data, labels)
    logging.info(f"Clustering Evaluation Metrics: {metrics}")


    # 6. Visualize Clusters
    visualizer = ClusterVisualizer(random_state=config.get('random_state', 42))
    vis_method = config.get('visualization_method', 'pca') # 'pca', 'tsne', 'svd'
    plot_title = f"{cluster_algo.capitalize()} Clustering of ArXiv Papers (k={n_clusters})"
    save_path = config.get('plot_save_path') # e.g., 'cluster_visualization.png'

    # Pass original vectorized data for reduction
    visualizer.plot(vectorized_data, labels, method=vis_method, title=plot_title, save_path=save_path)


    # Optional: Save clustered data
    output_csv_path = config.get('output_csv_path')
    if output_csv_path:
        try:
            # Select relevant columns to save
            df_output = df[['id', 'title', 'category', 'summary', 'processed_summary', 'cluster_label']]
            df_output.to_csv(output_csv_path, index=False)
            logging.info(f"Clustered data saved to {output_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save output CSV: {e}")


    end_time = time.time()
    logging.info(f"Pipeline finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    # --- Configuration ---
    # Ideally, load this from a YAML/JSON file or use argparse for command-line args
    pipeline_config = {
        "data_path": "arXiv_scientific dataset.csv",  # <--- IMPORTANT: SET YOUR FILE PATH HERE
        "sample_size": 2000,           # Use None for full dataset, or an integer for sampling
        "random_state": 42,
        "vectorizer_method": "tfidf",  # 'tfidf'
        "tfidf_max_features": 5000,
        "clustering_algorithm": "kmeans", # 'kmeans' or 'hierarchical'
        "n_clusters": 10,              # Desired number of clusters
        "hierarchical_linkage": "ward", # Use if algorithm is 'hierarchical' ('ward', 'average', 'complete')
        "evaluation_metrics": ["silhouette", "davies_bouldin"], # Metrics to compute
        "visualization_method": "tsne", # 'pca', 'tsne', 'svd'
        "plot_save_path": "arxiv_clusters_tsne.png", # Set path to save plot, or None to display
        "output_csv_path": "arxiv_clustered_output.csv" # Set path to save clustered data, or None
    }

    # --- Argument Parser (Example for command-line execution) ---
    parser = argparse.ArgumentParser(description="ArXiv Paper Clustering Pipeline")
    parser.add_argument("-f", "--file", type=str, default=pipeline_config["data_path"], help="Path to the input CSV dataset")
    parser.add_argument("-s", "--sample", type=int, default=pipeline_config["sample_size"], help="Number of samples to use (0 or negative for full dataset)")
    parser.add_argument("-k", "--clusters", type=int, default=pipeline_config["n_clusters"], help="Number of clusters (K)")
    parser.add_argument("-a", "--algo", type=str, default=pipeline_config["clustering_algorithm"], choices=['kmeans', 'hierarchical'], help="Clustering algorithm")
    parser.add_argument("-v", "--vis", type=str, default=pipeline_config["visualization_method"], choices=['pca', 'tsne', 'svd'], help="Visualization method")
    parser.add_argument("--saveplot", type=str, default=pipeline_config["plot_save_path"], help="File path to save the cluster plot (e.g., plot.png)")
    parser.add_argument("--savecsv", type=str, default=pipeline_config["output_csv_path"], help="File path to save the clustered output CSV (e.g., results.csv)")

    args = parser.parse_args()

    # Update config from command-line arguments
    pipeline_config["data_path"] = args.file
    pipeline_config["sample_size"] = args.sample if args.sample > 0 else None
    pipeline_config["n_clusters"] = args.clusters
    pipeline_config["clustering_algorithm"] = args.algo
    pipeline_config["visualization_method"] = args.vis
    pipeline_config["plot_save_path"] = args.saveplot
    pipeline_config["output_csv_path"] = args.savecsv


    # --- Run Pipeline ---
    run_pipeline(pipeline_config)