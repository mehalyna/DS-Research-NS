import requests
import json

# The URL of your local Django server's new endpoint
url = "http://127.0.0.1:8000/api/predict/state/"

# A dummy user profile payload
payload = {
    "Age": 21,
    "Gender": "Female",
    "Country": "Ukraine",
    "Coffee_Intake": 4,
    "Caffeine_mg": 450,
    "Sleep_Hours": 5.5,
    "BMI": 21.5,
    "Heart_Rate": 82,
    "Physical_Activity_Hours": 2.0,
    "Occupation": "Student",
    "Smoking": "No",
    "Alcohol_Consumption": "Yes"
}

headers = {"Content-Type": "application/json"}

print(f"Sending POST request to {url}...")

try:
    # Send the data to the server
    response = requests.post(url, json=payload, headers=headers)

    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("\n🎉 Success! Server responded with:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("\n❌ Failed! Server responded with:")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Connection failed. Is the Django server running in the other terminal?")