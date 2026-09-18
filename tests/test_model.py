import joblib
import pandas as pd

from src.config import MODEL_PATH


def test_model_exists():

    model = joblib.load(MODEL_PATH)

    assert model is not None