import requests
import json

# URL and key for the web service, should be similar to:
# 'http://17fb482f-c5d8-4df7-b011-9b33e860c111.southcentralus.azurecontainer.io/score'
scoring_uri = 'http://332fefcb-8ec5-44d5-98cf-36b8099e3115.southcentralus.azurecontainer.io/score'
key = 'G9YZdyl67xwpiAa6c8GEoB0sYAQxrPmL'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 75,
            "anaemia": 0,
            "creatinine_phosphokinase": 582,
            "diabetes": 0,
            "ejection_fraction": 20,
            "high_blood_pressure": 1,
            "platelets": 265000,
            "serum_creatinine": 1.9,
            "serum_sodium": 130,
            "sex": 1,
            "smoking": 0,
            "time": "4"
          },
          {
            "age": 55,
            "anaemia": 0,
            "creatinine_phosphokinase": 7861,
            "diabetes": 0,
            "ejection_fraction": 38,
            "high_blood_pressure": 0,
            "platelets": 263358.03,
            "serum_creatinine": 1.1,
            "serum_sodium": 136,
            "sex": 1,
            "smoking": 0,
            "time": "6"
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}

#If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


