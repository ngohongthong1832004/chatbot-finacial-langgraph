import uvicorn

if __name__ == "__main__":
    uvicorn.run("chatfinan.main:app", host="0.0.0.0", port=8005, reload=True) 