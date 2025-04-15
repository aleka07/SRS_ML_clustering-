
**4. How to Run**

1.  Make sure you have created the directory structure and all the files.
2.  Place the `arxivData.csv` file inside the `arxiv_clustering_lib` folder (or specify the correct path in `main.py` or via the command line).
3.  Navigate to the `arxiv_clustering_lib` directory in your terminal.
4.  Run the pipeline:
    *   With default settings (2000 samples, K-Means, k=10, t-SNE visualization):
        ```bash
        python main.py
        ```
    *   With custom settings (e.g., 5000 samples, Hierarchical clustering, k=8, PCA visualization, save plot and CSV):
        ```bash
        python main.py -s 5000 -a hierarchical -k 8 -v pca --saveplot hier_pca_k8.png --savecsv results_hier_k8.csv
        ```

The script will output logs to the console, display or save the cluster visualization plot, and optionally save a CSV file with the original data and the assigned cluster labels.