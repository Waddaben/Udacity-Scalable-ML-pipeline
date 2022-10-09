import os
import sys
import os
import inspect
import requests

from fastapi.testclient import TestClient

# add the starter directory to the path so we can import the train_model.py file
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200

def test_predict():
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
        }   
    )
    # print the response
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"prediction": "Income < 50k"}