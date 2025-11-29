import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics

def test_process_data_output_types():
    """
    Test that process_data returns the expected output types.
    """
    # tiny mock dataframe with the required columns
    df = pd.DataFrame({
        "age": [25, 45],
        "workclass": ["Private", "Self-emp"],
        "fnlgt": [100000, 200000],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Craft-repair"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

    X, y, encoder, lb = process_data(
        df,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label="salary",
        training=True
    )

    assert isinstance(X, pd.DataFrame) or hasattr(X, "shape")
    assert len(y) == 2
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_sklearn_model():
    """
    Ensure train_model returns an sklearn model with a predict method.
    """
    import numpy as np

    # Fake simple input
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)

    # Expects a model with .predict()
    assert hasattr(model, "predict")


def test_compute_model_metrics_returns_three_values():
    """
    Ensure compute_model_metrics returns precision, recall, and fbeta.
    """
    preds = [0, 1, 1, 0]
    labels = [0, 1, 0, 0]

    precision, recall, fbeta = compute_model_metrics(labels, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

