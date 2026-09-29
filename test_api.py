import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print(" Testing AI Voice Sales Agent API...")
    
    # Test 1: Start call
    print("\n1. Testing start call...")
    start_data = {
        "phone_number": "+1234567890",
        "customer_name": "Test Customer"
    }
    
    try:
        response = requests.post(f"{base_url}/start-call", json=start_data)
        if response.status_code == 200:
            call_data = response.json()
            call_id = call_data["call_id"]
            print(f" Call started: {call_id}")
            print(f"First message: {call_data['first_message']}")
        else:
            print(f" Failed to start call: {response.status_code}")
            return
    except Exception as e:
        print(f" Error starting call: {e}")
        return
    
    # Test 2: Send response
    print("\n2. Testing customer response...")
    response_data = {
        "message": "Tell me more about the course"
    }
    
    try:
        response = requests.post(f"{base_url}/respond/{call_id}", json=response_data)
        if response.status_code == 200:
            reply_data = response.json()
            print(f" Agent replied: {reply_data['reply']}")
        else:
            print(f" Failed to get response: {response.status_code}")
    except Exception as e:
        print(f" Error getting response: {e}")
    
    # Test 3: Get conversation
    print("\n3. Testing conversation history...")
    try:
        response = requests.get(f"{base_url}/conversation/{call_id}")
        if response.status_code == 200:
            conv_data = response.json()
            print(f" Conversation has {len(conv_data['history'])} messages")
            for msg in conv_data['history']:
                print(f"  {msg['sender']}: {msg['text'][:50]}...")
        else:
            print(f" Failed to get conversation: {response.status_code}")
    except Exception as e:
        print(f" Error getting conversation: {e}")
    
    print("\n API test completed!")

if __name__ == "__main__":
    test_api()
