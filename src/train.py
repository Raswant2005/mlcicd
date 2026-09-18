import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.logger import logger

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

from src.preprocessing_pipeline import (
    PreprocessingPipeline,
)


class ModelTrainer:

    def train(self, df: pd.DataFrame):

        logger.info("Training started...")

        X = df.drop(columns=["Churn"])

        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        preprocessor = (
            PreprocessingPipeline()
            .build()
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=RANDOM_STATE
                    ),
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )


        joblib.dump(
            pipeline,
            MODEL_PATH,
        )

        logger.info(
            "Model saved successfully."
        )

        return pipeline, X_test, y_test