from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.services.langgraph import process_query
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, Request
from dotenv import load_dotenv
load_dotenv()

import os
router = APIRouter()
oauth = OAuth()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SERVICE_AUTH= os.getenv("SERVICE_AUTH")
SERVICE_AUTH_URL= os.getenv("SERVICE_AUTH_URL")
oauth.register(
    name=SERVICE_AUTH,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url=SERVICE_AUTH_URL,
    client_kwargs={'scope': 'openid email profile'}
)
class Question(BaseModel):
    text: str = Field(..., description="The question to ask")

class Answer(BaseModel):
    answer: str = Field(..., description="The answer to the question")

# Google Login
@router.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")
    print("🔁 Redirect URI:", redirect_uri)
    return await oauth.google.authorize_redirect(request, redirect_uri)

# Google Auth Callback
@router.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = await oauth.google.userinfo(token=token) 
    print("✅ User Info:", user_info)
    frontend_url = f"http://localhost:3000?username={user_info['name']}&email={user_info['email']}"
    return RedirectResponse(url=frontend_url)

# RAG API
@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        result = process_query(question.text)
        return {"answer": result["generation"]}
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))