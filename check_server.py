import time
import requests

print("Waiting for server to start...")
time.sleep(3)

try:
    response = requests.get("http://localhost:8000")
    print(f" Server is running! Status: {response.status_code}")
    print(" Open http://localhost:8000 in your browser")
except Exception as e:
    print(f" Server not ready yet: {e}")
