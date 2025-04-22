from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chatfinan.api.routes import router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
app = FastAPI(
    title="LangGraph RAG API",
    description="API for RAG-based question answering system using LangGraph",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.mount("/static", StaticFiles(directory="chatfinan/static"), name="static")
app.include_router(router, prefix="/api")

@app.get("/")
async def read_root():
    return FileResponse("chatfinan/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatfinan.main:app", host="0.0.0.0", port=8005, reload=True) 