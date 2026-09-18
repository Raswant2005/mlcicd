from pathlib import Path

import pandas as pd

from app.logger import logger
from src.config import RAW_DATA


class DataIngestion:
    """
    Responsible for loading and validating raw data.
    """

    def __init__(self, filename: str):
        self.file_path = RAW_DATA / filename

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV into a DataFrame.
        """
        logger.info("Starting data ingestion...")

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        df = pd.read_csv(self.file_path)

        logger.info(
            "Dataset loaded successfully. Shape: %s",
            df.shape,
        )

        return df