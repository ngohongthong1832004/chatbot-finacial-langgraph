from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.services.langgraph import process_query
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import time
import psutil
import pkg_resources
load_dotenv()

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

# Health Check endpoints
@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "chatbot-langgraph-api",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    try:
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "chatbot-langgraph-api",
            "version": "1.0.0",
            "system": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024)
            },
            "environment": os.getenv("ENV", "development")
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.get("/health/db")
async def database_health_check():
    """Database connectivity health check"""
    try:
        # Import database health check function
        from src.database.health import test_connection
        
        # Test database connection
        db_status = test_connection()
        
        if db_status:
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": time.time()
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "database": "disconnected",
                    "timestamp": time.time()
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.get("/health/dependencies")
async def dependencies_health_check():
    """Check if all required dependencies are available"""
    try:
        dependencies = [
            "fastapi",
            "pydantic", 
            "langgraph",
            "psycopg2-binary",
            "authlib"
        ]
        
        installed_packages = {pkg.project_name.lower(): pkg.version for pkg in pkg_resources.working_set}
        
        dep_status = {}
        all_healthy = True
        
        for dep in dependencies:
            if dep.lower() in installed_packages:
                dep_status[dep] = {
                    "status": "installed",
                    "version": installed_packages[dep.lower()]
                }
            else:
                dep_status[dep] = {"status": "missing"}
                all_healthy = False
        
        status_code = 200 if all_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if all_healthy else "unhealthy",
                "dependencies": dep_status,
                "timestamp": time.time()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

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
    frontend_url = f"https://chatbot-finacial-langgraph.vercel.app/?username={user_info['name']}&email={user_info['email']}"
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