from fastapi import FastAPI, HTTPException

from app.model_lodder import ModelLoader
from app.predictor import Predictor
from app.schemas import PredictionRequest


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


loader = ModelLoader()

pipeline = loader.load()

predictor = Predictor()


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
    request: PredictionRequest
):

    try:

        result = predictor.predict(
            pipeline,
            request,
        )

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc