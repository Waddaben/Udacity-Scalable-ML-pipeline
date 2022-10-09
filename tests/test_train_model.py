"""
This file is used for testing the train_model.py file.
"""
import os
import inspect
import sys
import pytest
from sklearn.model_selection import train_test_split
import pandas as pd




# add the starter directory to the path so we can import the train_model.py file
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from starter.starter.ml.model import (
    train_and_test_on_slices,
    train_model,
    compute_model_metrics,
    inference,
)
from starter.starter.ml.data import process_data

# upload the census_cleaned.csv file
data_path = "starter/data/census_cleaned.csv"
data = pd.read_csv(data_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# prepare the data
X_data, y_data, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)


def test_train_model():
    """
    Test the model training
    """
    # test the train_model function
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=3
    )
    model = train_model(X_train, y_train, random_state=3)
    assert model is not None
    # assert a prediction
    # extract the first row of the numpy array data
    first_row = X_test[0]
    # shape the data to be in the correct format for the model
    first_row = first_row.reshape(1, -1)
    assert model.predict(first_row)[0] == 1
    assert y_test[0] == 1


def test_compute_metrics():
    # test the train_model function
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=10
    )
    model = train_model(X_train, y_train, random_state=10)
    predictions = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    assert precision >= 0.70
    assert recall >= 0.52
    assert fbeta >= 0.6


def test_inference():
    """
    Test the inference function
    """
    # test the train_and_test_on_slices function
    X_train, X_test, y_train, _ = train_test_split(
        X_data, y_data, test_size=0.2, random_state=10
    )
    model = train_model(X_train, y_train, random_state=10)
    assert all(model.predict(X_test)) == all(inference(model, X_test))
