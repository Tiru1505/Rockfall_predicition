import requests
import random
from datetime import datetime

url = "http://127.0.0.1:8000/predict"

for i in range(10):  # Insert 10 dummy records
    data = {
        "site_id": "mine-001",
        "slope_angle": random.uniform(20, 60),
        "rock_strength": random.uniform(30, 80),
        "joint_spacing": random.uniform(5, 20),
        "joint_orientation": random.uniform(0, 360),
        "water_content": random.uniform(5, 40),
        "temperature": random.uniform(15, 35),
        "humidity": random.uniform(30, 80),
        "wind_speed": random.uniform(0, 10),
        "rainfall": random.uniform(0, 20),
        "vibration_intensity": random.uniform(0, 1),
        "latitude": 12.9716,
        "longitude": 77.5946
    }
    response = requests.post(url, json=data)
    print(i+1, response.json())


{
  "site_id": "demo_site",
  "slope_angle": 85,
  "rock_strength": 5,
  "joint_spacing": 1,
  "joint_orientation": 45,
  "water_content": 90,
  "temperature": 50,
  "humidity": 95,
  "wind_speed": 40,
  "rainfall": 200,
  "vibration_intensity": 9,
  "latitude": -25.2744,
  "longitude": 133.7751
}
