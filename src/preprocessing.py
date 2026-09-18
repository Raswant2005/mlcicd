import pandas as pd

from app.logger import logger

class DataPreprocessor:
    
    def preprocess(self, df):

        logger.info("Starting preprocessing...")

        df = self.drop_customer_id(df)

        df = self.fix_total_charges(df)

        logger.info("Preprocessing completed.")

        return df
    
    def drop_customer_id(self, df):
    
        df = df.drop(columns=["customerID"])

        logger.info("customerID removed.")

        return df
    
    def fix_total_charges(self, df):
    
        df["TotalCharges"] = (
            df["TotalCharges"]
            .replace(" ", pd.NA)
        )

        df["TotalCharges"] = (
            pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )
        )

        df["TotalCharges"] = (
            df["TotalCharges"]
            .fillna(
                df["TotalCharges"].median()
            )
        )

        logger.info("TotalCharges cleaned.")

        return df