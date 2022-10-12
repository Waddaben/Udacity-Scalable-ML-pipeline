"""
This file is used for testing the test_api.py file.
"""
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    """
    This function test the get
    """
    request = client.get("/")
    assert request.status_code == 200
    assert request.json() == {"message": "Welcome to the starter project"}


def test_predict_1():
    """
    This function test the post
    """
    # get the client to make a post request to the /predict endpoint
    response = client.post(
        "/predict",
        json={
            "age": 28,
            "workclass": "Private",
            "fnlgt": 338409,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Wife",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 80,
            "native-country": "Cuba",
        },
    )
    # print the response
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"prediction": "Income < 50k"}


def test_predict_2():
    """
    This function test the post
    """
    # get the client to make a post request to the /predict endpoint
    response = client.post(
        "/predict",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    # print the response
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"prediction": "Income < 50k"}


def test_predict_3():
    """
    This function test the post
    """
    # get the client to make a post request to the /predict endpoint
    response = client.post(
        "/predict",
        json={
            "age": 50,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 83311,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-manageria",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 13,
            "native-country": "United-States",
        },
    )
    # print the response
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"prediction": "Income < 50k"}
