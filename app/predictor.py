import pandas as pd


class Predictor:

    def predict(
        self,
        pipeline,
        request,
    ):

        data = request.model_dump()

        df = pd.DataFrame([data])

        prediction = pipeline.predict(df)[0]

        probability = pipeline.predict_proba(df)[0]

        classes = pipeline.classes_

        probabilities = dict(
            zip(classes, probability)
        )

        return {
            "prediction": prediction,
            "probability": float(
                probabilities[prediction]
            ),
        }