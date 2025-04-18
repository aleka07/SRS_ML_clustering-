# visualize_3d_interactive.py

import argparse
import logging
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.decomposition import PCA, TruncatedSVD
import plotly.express as px
import plotly.graph_objects as go # Для более тонкой настройки

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reduce_dimensions_3d(data, method='svd', random_state=42):
    """
    Reduces data dimensionality to 3 components.

    Args:
        data: Sparse or dense data matrix.
        method (str): 'pca' or 'svd'. SVD is generally better for sparse TF-IDF.
        random_state (int): Random seed.

    Returns:
        np.ndarray: Data reduced to 3 dimensions, or None on failure.
    """
    n_components = 3
    logging.info(f"Reducing dimensionality to {n_components}D using {method.upper()}...")
    try:
        if method == 'pca':
            # PCA might require dense conversion, prefer SVD for sparse
            if hasattr(data, 'toarray'): # Check if sparse
                 logging.warning("Using PCA on sparse data requires conversion to dense. This might use a lot of memory. Consider using 'svd'.")
                 data_dense = data.toarray()
            else:
                 data_dense = data
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_data = reducer.fit_transform(data_dense)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
            reduced_data = reducer.fit_transform(data)
        else:
            logging.error(f"Unsupported reduction method: {method}. Choose 'pca' or 'svd'.")
            return None
        logging.info(f"Dimensionality reduction complete. Output shape: {reduced_data.shape}")
        return reduced_data
    except MemoryError:
        logging.error(f"MemoryError during {method.upper()} dimensionality reduction. Input shape: {data.shape}. Try reducing sample size or using 'svd'.")
        return None
    except Exception as e:
        logging.error(f"Error during {method.upper()} dimensionality reduction: {e}")
        return None

def plot_interactive_3d(df_plot, title="Interactive 3D Cluster Visualization", output_html="interactive_3d_plot.html"):
    """
    Creates and saves/shows an interactive 3D scatter plot using Plotly.

    Args:
        df_plot (pd.DataFrame): DataFrame with columns 'x', 'y', 'z', 'cluster_label', and 'hover_text'.
        title (str): Title for the plot.
        output_html (str | None): Path to save the HTML file. If None, displays the plot.
    """
    if not all(col in df_plot.columns for col in ['x', 'y', 'z', 'cluster_label', 'hover_text']):
        logging.error("DataFrame for plotting is missing required columns ('x', 'y', 'z', 'cluster_label', 'hover_text').")
        return

    logging.info("Generating interactive 3D plot...")
    try:
        fig = px.scatter_3d(
            df_plot,
            x='x',
            y='y',
            z='z',
            color='cluster_label',       # Color points by cluster
            hover_name='hover_text',     # Text shown prominently on hover
            # hover_data=['id'],         # Additional data on hover (optional)
            opacity=0.7,                 # Make points slightly transparent
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3', 'cluster_label': 'Cluster'},
            color_discrete_sequence=px.colors.qualitative.Vivid # Use a distinct color scheme
        )

        # Optional: Customize layout further
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40), # Reduce margins
            legend_title_text='Cluster'
        )
        fig.update_traces(marker=dict(size=3)) # Adjust marker size

        if output_html:
            fig.write_html(output_html)
            logging.info(f"Interactive 3D plot saved to {output_html}")
        else:
            fig.show() # Opens in a web browser

    except Exception as e:
        logging.error(f"Failed to create/save interactive 3D plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Interactive 3D Cluster Visualization")
    parser.add_argument("csv_path", help="Path to the CSV file containing clustered data (output from main.py)")
    parser.add_argument("vectors_path", help="Path to the saved vectorized data (.npz file)")
    parser.add_argument("labels_path", help="Path to the saved cluster labels (.npy file)")
    parser.add_argument("-m", "--method", default="svd", choices=['pca', 'svd'], help="Dimensionality reduction method (default: svd)")
    parser.add_argument("-o", "--output", default="interactive_3d_clusters.html", help="Output HTML file path (default: interactive_3d_clusters.html). If 'None' or empty, shows plot instead of saving.")
    parser.add_argument("-t", "--title", default="Interactive 3D Cluster Visualization", help="Title for the plot")

    args = parser.parse_args()

    output_file = args.output if args.output and args.output.lower() != 'none' else None

    # 1. Load Data
    logging.info("Loading data...")
    try:
        df_data = pd.read_csv(args.csv_path)
        vectors = load_npz(args.vectors_path)
        labels = np.load(args.labels_path)
        logging.info(f"Loaded data: CSV shape={df_data.shape}, Vectors shape={vectors.shape}, Labels shape={labels.shape}")

        # Basic validation
        if not (df_data.shape[0] == vectors.shape[0] == len(labels)):
             raise ValueError("Mismatch in number of samples between CSV, vectors, and labels.")
        if 'cluster_label' not in df_data.columns:
            logging.warning("Column 'cluster_label' not found in CSV, using loaded labels array.")
            # Ensure labels from file match loaded df length
            if df_data.shape[0] == len(labels):
                 df_data['cluster_label'] = labels
            else:
                 # This case should be handled by the earlier check, but being defensive
                 raise ValueError("Cannot align loaded labels with CSV data due to length mismatch.")
        elif not np.array_equal(df_data['cluster_label'].values, labels):
             logging.warning("Labels in CSV differ from loaded .npy labels. Using labels from CSV.")
             labels = df_data['cluster_label'].values # Prioritize CSV if present and matches length

    except FileNotFoundError as e:
        logging.error(f"Error loading files: {e}. Make sure paths are correct.")
        exit(1)
    except Exception as e:
        logging.error(f"An error occurred during data loading: {e}")
        exit(1)

    # 2. Reduce Dimensions
    reduced_data = reduce_dimensions_3d(vectors, method=args.method)

    if reduced_data is None:
        logging.error("Halting due to dimensionality reduction failure.")
        exit(1)

    # 3. Prepare DataFrame for Plotly
    df_plot = pd.DataFrame(reduced_data, columns=['x', 'y', 'z'], index=df_data.index) # Keep index alignment
    df_plot['cluster_label'] = labels.astype(str) # Plotly prefers categorical data for colors

    # Create hover text (customize as needed)
    df_plot['hover_text'] = df_data['id'] + ': ' + df_data['title'].str.slice(0, 100) + '...' # ID + Title snippet
    df_plot['id'] = df_data['id'] # Add ID if needed for hover_data


    # 4. Plot
    plot_interactive_3d(df_plot, title=args.title, output_html=output_file)

    logging.info("3D Visualization script finished.")