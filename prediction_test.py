import requests
import json
import fastapi

url = "http://127.0.0.1:8000/predict"
data = {
    "age": 40,
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

# request a response from the API using FastAPI
response = requests.post(url, json=data)

# print the response
print(response.json())