import json
import os

import requests

FILE_PATH = ".data/sample_regression_data.csv"

endpoint = "your_endpoint_url"
token = os.getenv("DYNAMIQ_ACCESS_KEY")

streaming = False

headers = {
    "Authorization": f"Bearer {token}",
}

# Payload: Modify input schema as per the input node schema defined in the UI
payload = {
    "input": {"input": "Show me columns"},
    "stream": streaming,
}

# Make a POST request to the deployed endpoint
response = requests.post(
    endpoint,
    data={"input": json.dumps(payload["input"])},
    files={"files": open(FILE_PATH, "rb")},
    headers=headers,
    stream=streaming,
    timeout=300,
)

# Print the response
print(response.json())
