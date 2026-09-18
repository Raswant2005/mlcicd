import pandas as pd

from app.logger import logger


class DataValidation:

    REQUIRED_COLUMNS = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    def validate(self, df: pd.DataFrame):
        logger.info("Starting validation...")

        self.check_empty(df)
        self.check_columns(df)
        self.check_duplicates(df)

        logger.info("Validation completed.")

    def check_empty(self, df):
        if df.empty:
            raise ValueError("Dataset is empty.")

        logger.info("Dataset is not empty.")

    def check_columns(self, df):
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        logger.info("All required columns are present.")

    def check_duplicates(self, df):
        duplicates = df.duplicated().sum()

        logger.info(
            "Duplicate rows: %d",
            duplicates
        )