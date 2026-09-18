import pandas as pd

from src.preprocessing import DataPreprocessor


def test_customer_id_removed():

    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "TotalCharges": ["100", "200"]
    })

    preprocessor = DataPreprocessor()

    result = preprocessor.preprocess(df)

    assert "customerID" not in result.columns
    
    
def test_total_charges_converted_to_numeric():
    
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "TotalCharges": ["100", "200"]
    })

    preprocessor = DataPreprocessor()

    result = preprocessor.preprocess(df)

    assert result["TotalCharges"].dtype != "object"