import joblib

from app.logger import logger

from src.config import MODEL_PATH


class ModelLoader:

    def __init__(self):

        self.pipeline = None

    def load(self):

        if self.pipeline is None:

            logger.info(
                "Loading trained model..."
            )

            self.pipeline = joblib.load(
                MODEL_PATH
            )

        return self.pipeline