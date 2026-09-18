"""
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def main():

    ingestion = DataIngestion(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df = ingestion.load_data()

    validator = DataValidation()

    validator.validate(df)

    preprocessor = DataPreprocessor()

    df = preprocessor.preprocess(df)

    trainer = ModelTrainer()

    X_test, y_test = trainer.train(df)

    evaluator = ModelEvaluator()

    metrics = evaluator.evaluate(
        X_test,
        y_test,
    )

    print(metrics)


if __name__ == "__main__":
    main()
"""
    
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.preprocessing import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.mlfllow_tracker import MLFlowTracker


def main():

    # Data Ingestion
    ingestion = DataIngestion(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    df = ingestion.load_data()

    # Data Validation
    validator = DataValidation()
    validator.validate(df)

    # Data Preprocessing
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df)

    # Model Training
    trainer = ModelTrainer()
    pipeline, X_test, y_test = trainer.train(df)

    # Model Evaluation
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(
        X_test,
        y_test,
    )

    # MLflow Logging
    tracker = MLFlowTracker()
    tracker.log_run(
        pipeline=pipeline,
        metrics=metrics,
        params={
            "model": "RandomForest",
            "random_state": 42,
        },
    )

    print(metrics)


if __name__ == "__main__":
    main()