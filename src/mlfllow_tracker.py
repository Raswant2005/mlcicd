import mlflow
import mlflow.sklearn

from app.logger import logger


class MLFlowTracker:

    def log_run(
        self,
        pipeline,
        metrics,
        params,
    ):

        logger.info(
            "Logging to MLflow..."
        )

        mlflow.set_experiment(
            "Customer Churn Prediction"
        )

        with mlflow.start_run():

            for key, value in params.items():

                mlflow.log_param(
                    key,
                    value,
                )

            for key, value in metrics.items():

                mlflow.log_metric(
                    key,
                    value,
                )

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
            )

        logger.info(
            "MLflow logging completed."
        )