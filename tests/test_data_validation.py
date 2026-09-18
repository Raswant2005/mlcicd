import pandas as pd
import pytest

from src.data_validation import DataValidation


def test_dataset_not_empty():
    df = pd.DataFrame({
        "customerID": ["001"],
        "gender": ["Male"],
        "Churn": ["Yes"]
    })

    validator = DataValidation()

    validator.check_empty(df)


def test_empty_dataset():
    df = pd.DataFrame()

    validator = DataValidation()

    with pytest.raises(ValueError):
        validator.check_empty(df)


def test_required_columns():
    df = pd.DataFrame({
        "customerID": ["001"],
        "gender": ["Male"],
        "Churn": ["Yes"]
    })

    validator = DataValidation()

    with pytest.raises(ValueError):
        validator.check_columns(df)