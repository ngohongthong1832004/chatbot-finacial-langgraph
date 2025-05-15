# LangGraph RAG API

Đây là API FastAPI cho hệ thống RAG (Retrieval Augmented Generation) kết hợp truy xuất vector, truy vấn cơ sở dữ liệu SQL và tìm kiếm web để trả lời câu hỏi sử dụng LangGraph.

## Tính năng

- Truy xuất tài liệu dựa trên vector
- Tích hợp cơ sở dữ liệu SQL
- Tìm kiếm web
- Sinh câu trả lời thông minh
- Workflow linh hoạt với LangGraph

## Hướng dẫn cài đặt

### Cách 1: Chạy bằng Docker Compose (Khuyên dùng)

1. **Tạo file `.env` trong `API-chatbot-langgraph/`** với nội dung:
   ```env
   # Key LLM và key Tavily
   OPENAI_API_KEY=your-openai-api-key-here
   TAVILY_API_KEY=your-tavily-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here

   # Database Configuration
   DBUSER=your-db-username-here
   DBPASSWORD=your-db-password-here
   DBHOST=your-db-host-here
   DBPORT=your-db-port-here
   DBNAME=your-db-name-here
   DBDATABASE=your-db-database-here

   # GGoogle OAuth Configuration
   CLIENT_ID=your-client-id-here
   CLIENT_SECRET=your-client-secret-here
   SERVICE_AUTH=google
   SERVICE_AUTH_URL=https://accounts.google.com/.well-known/openid-configuration
   ```

2. **Build và chạy toàn bộ hệ thống:**
   ```bash
   docker compose up --build
   ```
   - Backend: http://localhost:8000
   - Frontend: http://localhost:3000

> **Lưu ý:** Đảm bảo các file vector (`faiss_index.bin`, `embeddings.pkl`, `chunks.json`) đã có trong `src/services/sec_embeddings/`.

---

### Cách 2: Cài đặt thủ công bằng Python

1. **Tạo môi trường ảo:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Tạo file `.env`** như hướng dẫn trên.

4. **Chạy ứng dụng:**
   ```bash
   uvicorn src.main:app --reload
   ```
   - API docs: http://localhost:8000/docs

---

## Sử dụng API

Gửi POST tới `/api/ask` với JSON:
```json
{
    "text": "Câu hỏi của bạn"
}
```

Ví dụ với curl:
```bash
curl -X POST "http://localhost:8000/api/ask" -H "Content-Type: application/json" -d '{"text":"Thủ đô của Pháp là gì?"}'
```

## Tài liệu API

Sau khi chạy server, truy cập Swagger UI tại:
```
http://localhost:8000/docs
```

## Cấu trúc thư mục

```
API-chatbot-langgraph/               # Backend FastAPI (RAG agentic)
├── src/
│   ├── api/                         # API routes
│   ├── database/                    # Kết nối và truy vấn DB
│   │   └── connection.py            # Kết nối DB Supabase
│   ├── metadata/                    # Giới thiệu metadata của database
│   ├── services/                    # Logic RAG, vector search
│   │   └── sec_embeddings/          # Dữ liệu vector hóa (faiss_index.bin, embeddings.pkl, chunks.json)
│   │   └── langgraph.py             # Core LangGraph, agentic
│   └── main.py                      # Entrypoint FastAPI
├── .env.example                     # File mẫu .env
├── requirements.txt                 # Thư viện Python
├── Dockerfile                       # Dockerfile backend
├── run.py                           # Chạy FastAPI
└── README.md
```

## Quy trình xử lý

1. Người dùng gửi câu hỏi
2. Hệ thống truy xuất tài liệu liên quan từ vector DB
3. Đánh giá mức độ liên quan
4. Nếu là truy vấn DB → sinh SQL và truy vấn
5. Nếu không liên quan → tìm kiếm web
6. Nếu liên quan → sinh câu trả lời
7. Trả kết quả cho người dùng 