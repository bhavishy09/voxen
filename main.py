from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models import CallStart, CallResponse, Call, CallHistory
from llm_service import LLMService
from voice_service import VoiceService
import uuid
from datetime import datetime
from typing import Dict

app = FastAPI(title="AI Voice Sales Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
calls_db: Dict[str, Call] = {}

# Services
llm_service = LLMService()
voice_service = VoiceService()

@app.post("/start-call")
async def start_call(call_data: CallStart):
    """Start a new call session"""
    call_id = str(uuid.uuid4())
    
    first_message = f"Hi {call_data.customer_name}, this is your AI assistant calling about our AI Mastery Bootcamp. Can I share a quick detail with you?"
      # Create new call
    call = Call(
        call_id=call_id,
        customer_name=call_data.customer_name,
        phone_number=call_data.phone_number,
        start_time=datetime.now(),
        history=[CallHistory(
            sender="agent",
            text=first_message,
            timestamp=datetime.now()
        )]
    )
    
    calls_db[call_id] = call
      # Play the first message
    if not voice_service.text_to_speech(first_message):
        voice_service.fallback_tts(first_message)
    
    return {
        "call_id": call_id,
        "message": f"Calling {call_data.customer_name}...",
        "first_message": first_message
    }

@app.post("/respond/{call_id}")
async def respond_to_call(call_id: str, response: CallResponse):
    """Process customer response and generate reply using custom rules"""
    if call_id not in calls_db:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call = calls_db[call_id]
    
    if not call.is_active:
        raise HTTPException(status_code=400, detail="Call has ended")
    
    # Add customer message to history
    customer_history = CallHistory(
        sender="customer",
        text=response.message,
        timestamp=datetime.now()
    )
    call.history.append(customer_history)
    
    # Generate AI response using fallback rules
    ai_reply = llm_service._get_fallback_response(response.message)
    should_end = llm_service.should_end_call(response.message)
    
    # Add AI response to history
    agent_history = CallHistory(
        sender="agent",
        text=ai_reply,
        timestamp=datetime.now()
    )
    call.history.append(agent_history)
    
    if should_end:
        call.is_active = False
    
    # Play AI response
    voice_service.text_to_speech(ai_reply)
    
    return {
        "reply": ai_reply,
        "should_end_call": should_end
    }

@app.post("/respond-rag/{call_id}")
async def respond_to_call_rag(call_id: str, response: CallResponse):
    """Process customer response and generate reply using RAG system"""
    if call_id not in calls_db:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call = calls_db[call_id]
    
    if not call.is_active:
        raise HTTPException(status_code=400, detail="Call has ended")
    
    # Add customer message to history
    customer_history = CallHistory(
        sender="customer",
        text=response.message,
        timestamp=datetime.now()
    )
    call.history.append(customer_history)
      # Generate AI response using RAG system
    ai_reply = llm_service.get_rag_response(response.message)
    should_end = llm_service.should_end_call(response.message)
    
    # Add AI response to history
    agent_history = CallHistory(
        sender="agent",
        text=ai_reply,
        timestamp=datetime.now()
    )
    call.history.append(agent_history)
    
    if should_end:
        call.is_active = False
    
    # Play AI response
    voice_service.text_to_speech(ai_reply)
    
    return {
        "reply": ai_reply,
        "should_end_call": should_end
    }

@app.get("/conversation/{call_id}")
async def get_conversation(call_id: str):
    """Get full conversation history"""
    if call_id not in calls_db:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call = calls_db[call_id]
    
    history = [
        {
            "sender": h.sender,
            "text": h.text,
            "timestamp": h.timestamp.isoformat()
        }
        for h in call.history
    ]
    
    return {
        "call_id": call_id,
        "customer_name": call.customer_name,
        "phone_number": call.phone_number,
        "is_active": call.is_active,
        "history": history
    }

@app.get("/simulate-call/{call_id}")
async def simulate_call(call_id: str):
    """Simulate a voice call with speech recognition"""
    if call_id not in calls_db:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call = calls_db[call_id]
    
    if not call.is_active:
        return {"message": "Call has ended", "customer_said": "Error occurred"}
    
    print(" Starting speech recognition...")
    
    # Listen for customer response with longer timeout
    customer_speech = voice_service.speech_to_text(timeout=5)
    
    print(f" Speech result: {customer_speech}")
    
    # Always return the speech result, even if it's an error
    if customer_speech in ["No response", "Could not understand", "Error occurred"]:
        return {
            "message": "Could not capture speech", 
            "customer_said": customer_speech,
            "speech": customer_speech
        }
    
    # Process the response
    response = CallResponse(message=customer_speech)
    result = await respond_to_call(call_id, response)
    
    return {
        "customer_said": customer_speech,
        "agent_replied": result["reply"],
        "should_end_call": result["should_end_call"]
    }

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/api")
async def api_info():
    return {"message": "AI Voice Sales Agent API", "docs": "/docs"}

@app.get("/rag-status")
async def get_rag_status():
    """Get RAG system status"""
    return llm_service.get_rag_status()

@app.post("/rag-respond/{call_id}")
async def rag_respond_to_call(call_id: str, response: CallResponse):
    """Process customer response using RAG system only"""
    if call_id not in calls_db:
        raise HTTPException(status_code=404, detail="Call not found")
    
    call = calls_db[call_id]
    
    if not call.is_active:
        raise HTTPException(status_code=400, detail="Call has ended")
    
    # Add customer message to history
    customer_history = CallHistory(
        sender="customer",
        text=response.message,
        timestamp=datetime.now()
    )
    call.history.append(customer_history)
    
    # Generate RAG response
    ai_reply = llm_service.generate_rag_response(response.message)
    should_end = llm_service.should_end_call(response.message)
      # Add AI response to history
    agent_history = CallHistory(
        sender="agent",
        text=ai_reply,
        timestamp=datetime.now()
    )
    call.history.append(agent_history)
    
    if should_end:
        call.is_active = False
        call.end_time = datetime.now()
    
    # Play AI response
    voice_service.text_to_speech(ai_reply)
    
    return {
        "reply": ai_reply,
        "should_end_call": should_end
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
