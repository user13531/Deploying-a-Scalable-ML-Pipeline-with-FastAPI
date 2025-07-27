import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url = "http://127.0.0.1:8000"
r = requests.get(url) # Your code here

# TODO: print the status code
print("Status Code:", r.status_code)
# TODO: print the welcome message
print("Response:", r.json()) 

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
url = "http://127.0.0.1:8000/data"
r = requests.post(url, json=data) # Your code here

# TODO: print the status code
print("Status Code:", r.status_code)
# TODO: print the result
print("Response:", r.json())
# The result should be the predicted salary class (e.g., ">50K" or "<)
