# data_loader/loader.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Handles loading data from a CSV file.
    """
    def __init__(self):
        pass

    def load_data(self, filepath: str) -> pd.DataFrame | None:
        """
        Loads data from the specified CSV file path.

        Args:
            filepath (str): The path to the CSV file.

        Returns:
            pd.DataFrame | None: A pandas DataFrame containing the loaded data,
                                 or None if loading fails.
        """
        try:
            logging.info(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)
            # Basic validation
            if 'summary' not in df.columns or 'id' not in df.columns:
                logging.error("Required columns ('id', 'summary') not found in CSV.")
                return None
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logging.error(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            logging.error(f"An error occurred during data loading: {e}")
            return None

    def get_sample(self, df: pd.DataFrame, n_samples: int | None = None, random_state: int = 42) -> pd.DataFrame:
        """
        Returns a random sample of the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            n_samples (int | None): The number of samples to retrieve. If None, returns the full DataFrame.
            random_state (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: A sampled DataFrame.
        """
        if n_samples is None or n_samples >= len(df):
            logging.info("Using full dataset.")
            return df
        elif n_samples > 0:
            logging.info(f"Sampling {n_samples} records from the dataset.")
            return df.sample(n=n_samples, random_state=random_state)
        else:
            logging.warning("n_samples must be a positive integer. Returning full dataset.")
            return df