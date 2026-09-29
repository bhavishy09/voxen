from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class CallStart(BaseModel):
    phone_number: str
    customer_name: str

class CallResponse(BaseModel):
    message: str

class CallHistory(BaseModel):
    sender: str  # "agent" or "customer"
    text: str
    timestamp: datetime

class Call(BaseModel):
    call_id: str
    customer_name: str
    phone_number: str
    history: List[CallHistory] = []
    is_active: bool = True
    start_time: datetime = datetime.now()
    end_time: Optional[datetime] = None
