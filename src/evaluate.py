import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from app.logger import logger

from src.config import MODEL_PATH


class ModelEvaluator:

    def evaluate(
        self,
        X_test,
        y_test,
    ):

        logger.info(
            "Evaluating model..."
        )

        pipeline = joblib.load(
            MODEL_PATH
        )

        predictions = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label="Yes",
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label="Yes",
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label="Yes",
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        report = classification_report(
            y_test,
            predictions,
        )

        logger.info(
            "Accuracy : %.3f",
            accuracy,
        )

        logger.info(
            "Precision : %.3f",
            precision,
        )

        logger.info(
            "Recall : %.3f",
            recall,
        )

        logger.info(
            "F1 Score : %.3f",
            f1,
        )

        logger.info(
            "\nConfusion Matrix\n%s",
            matrix,
        )

        logger.info(
            "\n%s",
            report,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }