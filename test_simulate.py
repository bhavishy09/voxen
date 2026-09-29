#!/usr/bin/env python3
"""Test the simulate voice endpoint"""

import requests
import json

def test_simulate_endpoint():
    print("Testing simulate voice endpoint...")
    
    # First, start a call
    start_data = {
        "customer_name": "Test User",
        "phone_number": "123-456-7890"
    }
    
    print("1. Starting a call...")
    start_response = requests.post("http://localhost:8001/start-call", json=start_data)
    
    if start_response.status_code == 200:
        call_data = start_response.json()
        call_id = call_data["call_id"]
        print(f" Call started with ID: {call_id}")
        
        # Now test the simulate endpoint
        print("\n2. Testing simulate voice endpoint...")
        print("   This will try to capture your voice...")
        
        simulate_response = requests.get(f"http://localhost:8001/simulate-call/{call_id}")
        
        if simulate_response.status_code == 200:
            result = simulate_response.json()
            print(" Simulate endpoint response:")
            print(json.dumps(result, indent=2))
        else:
            print(f" Simulate endpoint failed: {simulate_response.status_code}")
            print(f"Error: {simulate_response.text}")
    else:
        print(f" Failed to start call: {start_response.status_code}")
        print(f"Error: {start_response.text}")

if __name__ == "__main__":
    test_simulate_endpoint()
