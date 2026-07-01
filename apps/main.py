from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import time

from apps.chatbot import SHLChatbot

app = FastAPI(title="SHL AI Hiring Assistant")


chatbot = SHLChatbot()



class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]



@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    start_time = time.time()

    try:
        messages = [m.dict() for m in request.messages]

        
        result = chatbot.chat(messages)

        
        response = {
            "reply": result.get("reply", ""),
            "recommendations": result.get("recommendations", []),
            "end_of_conversation": result.get("end_of_conversation", False),
        }

       

        # Ensure recommendations is a list
        if not isinstance(response["recommendations"], list):
            response["recommendations"] = []

        
        response["recommendations"] = response["recommendations"][:10]

        cleaned_recommendations = []

        for rec in response["recommendations"]:
            if isinstance(rec, dict):
                cleaned_recommendations.append({
                    "name": rec.get("name", ""),
                    "url": rec.get("url", ""),
                    "test_type": rec.get("test_type", "")
                })

        response["recommendations"] = cleaned_recommendations

        
        response["end_of_conversation"] = bool(response["end_of_conversation"])

        
        return response

    except Exception as e:
       
        return {
            "reply": "Sorry, I encountered an error while processing your request.",
            "recommendations": [],
            "end_of_conversation": False,
        }