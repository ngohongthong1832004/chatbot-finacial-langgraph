@echo off
echo === Starting LangGraph Chatbot API ===

REM Di chuyển tới thư mục chứa project
cd .\API-chatbot-langgraph\

REM (Tùy chọn) Kích hoạt virtual environment nếu có
REM call venv\Scripts\activate

REM Chạy file Python chính
python run.py

pause
