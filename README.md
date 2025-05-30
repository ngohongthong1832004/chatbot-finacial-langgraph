# CHATBOT-LANGGRAPH

## DEMO: [chatbot-finacial-langgraph.vercel.app](https://chatbot-finacial-langgraph.vercel.app/)

## Giới thiệu
Cùng với sự phát triển của công nghệ AI và Agentic, việc xây dựng một hệ thống Chatbot thông minh, tự động trả lời câu hỏi từ người dùng về tài chính đang trở thành một xu hướng quan trọng. Dự án này nhằm mục đích phát triển một hệ thống Chatbot sử dụng công nghệ RAG (Retrieval-Augmented Generation) để cải thiện độ chính xác và khả năng truy xuất thông tin từ nhiều nguồn dữ liệu khác nhau. Có thể tương tác vơi database SQL, tìm kiếm web, sinh câu trả lời tự động bằng LLM (Large Language Model) như OpenAI, Google Gemini. Hệ thống này sẽ hỗ trợ nhiều tính năng như tìm kiếm tài liệu, truy vấn cơ sở dữ liệu, tìm kiếm web và sinh câu trả lời tự động. Hệ thống có thể mở rộng và tùy chỉnh dễ dàng cho nhiều lĩnh vực khác nhau.

## Lý do dự án
- Hệ thống Chatbot thông minh, tự động trả lời câu hỏi từ người dùng về tài chính.
- Sử dụng công nghệ RAG (Retrieval-Augmented Generation) để cải thiện độ chính xác và khả năng truy xuất thông tin.
- Kết hợp nhiều nguồn dữ liệu: tài liệu vector, truy vấn SQL, tìm kiếm web, sinh câu trả lời bằng LLM.
- Hỗ trợ nhiều tính năng như tìm kiếm tài liệu, truy vấn cơ sở dữ liệu, tìm kiếm web, sinh câu trả lời tự động.
- Hệ thống có thể mở rộng và tùy chỉnh dễ dàng cho nhiều lĩnh vực khác nhau.
- Hệ thống có thể tích hợp với các dịch vụ bên ngoài như OpenAI, Google Gemini, Tavily để nâng cao khả năng sinh câu trả lời và tìm kiếm thông tin.

## Tổng quan dự án

Hệ thống Chatbot Agentic RAG (Retrieval-Augmented Generation) thông minh, trả lời tự động dựa trên:
- Truy xuất tài liệu vector (FAISS)
- Truy vấn SQL (PostgreSQL)
- Tìm kiếm web (Tavily)
- Sinh câu trả lời bằng LLM (OpenAI, Google Gemini)
- Workflow agentic linh hoạt với LangGraph
- Giao diện React hiện đại, hỗ trợ markdown/code, đăng nhập Google
- Pipeline xử lý dữ liệu tài chính DJIA, embedding.

## Kiến trúc hệ thống
![Kiến trúc hệ thống](architecture/architecture.png)


## Cấu trúc thư mục tổng thể

```
project_root/
├── API-chatbot-langgraph/               # Backend FastAPI (RAG agentic)
│   ├── src/
│   │   ├── api/                         # API routes
│   │   ├── database/                    # Kết nối và truy vấn DB
│   │   │   └── connection.py            # Kết nối DB Supabase
│   │   ├── metadata/                    # Giới thiệu metadata của database
│   │   ├── services/                    # Logic RAG, vector search
│   │   │   └── sec_embeddings/          # Dữ liệu vector hóa (faiss_index.bin, embeddings.pkl, chunks.json)
│   │   │   └── langgraph.py             # Core LangGraph, agentic
│   │   └── main.py                      # Entrypoint FastAPI
│   ├── .env.example                     # File mẫu .env
│   ├── requirements.txt                 # Thư viện Python
│   ├── Dockerfile                       # Dockerfile backend
│   ├── run.py                           # Chạy FastAPI
│   └── README.md
├── architecture/                        # Kiến trúc hệ thống
├── Chatbot-FE/                          # Frontend React
│   ├── public/
│   ├── src/
│   │   └── App.css/                     # CSS cho ứng dụng
│   │   └── App.js/                      # Component chính của ứng dụng
│   │   └── index.css/                   # CSS cho ứng dụng
│   │   └── index.js/                    # Entry point của ứng dụng
│   │   └── reportWebVitals.js/           
│   ├── package.json                     # File cấu hình npm
│   ├── Dockerfile                       # Dockerfile frontend
│   └── README.md
├── final-data/                          # Pipeline xử lý dữ liệu tài chính, mapping, embedding
│   ├── flattened_sec_data/              # Dữ liệu đã xử lý, chunk, jsonl
│   ├── sec_embeddings/                  # Vector hóa dữ liệu tài chính
│   ├── download_djia_stock_prices.py    # Script crawl giá cổ phiếu
│   ├── download_djia_companies.py       # Script crawl thông tin công ty
│   ├── create_embeddings.py             # Script tạo embedding
│   ├── readme.htm                       # Kí hiệu báo cáo tài chính
│   ├── flatten.py                       # Script flatten dữ liệu
│   ├── test_rag.py                      # Test Retrieval-Augmented Generation
│   ├── README.md
│   └── ...
├── data/                                # Dữ liệu test (pdf, csv, tài chính, phương pháp)
├── dum-data/                            # Dữ liệu mẫu, thử nghiệm
├── test-components/                     # Notebook, script test pipeline, demo agent, SQL agent
│   ├── langgraph.ipynb                  # Demo pipeline LangGraph
│   ├── sql_agent_implementation.py      # Test SQL agent
│   └── ...
├── docker-compose.yml                   # Chạy fullstack FE + BE
└── .gitignore                           # File ignore git
└── README.md                            # File hướng dẫn tổng này
└── run.bat                              # File chạy nhanh trên Windows
```


## Yêu cầu hệ thống
- Python >= 3.8
- Node.js >= 18
- PostgreSQL >= 12
- Docker (nếu chạy bằng Docker)
- Docker Compose (nếu chạy bằng Docker Compose)
- Các thư viện Python trong `requirements.txt`
- Các thư viện Node.js trong `package.json`
- Các API keys cho OpenAI, Gemini, Tavily (nếu sử dụng)
- Cấu hình OAuth cho Google (nếu sử dụng đăng nhập Google)
- Cấu hình CORS cho backend (nếu sử dụng frontend khác domain)
- Cấu hình database Supabase (nếu sử dụng)
- Cấu hình vector DB (nếu sử dụng FAISS)
- Cấu hình môi trường (nếu sử dụng Docker hoặc Docker Compose)
- Cấu hình môi trường ảo (nếu không sử dụng Docker)


## Cài đặt môi trường
### 1. Cài đặt Python và Node.js
- Cài đặt Python >= 3.8 từ [python.org](https://www.python.org/downloads/)
- Cài đặt Node.js >= 18 từ [nodejs.org](https://nodejs.org/en/download/)

### 2. clone repo
```bash
git clone https://github.com/ngohongthong1832004/chatbot-finacial-langgraph.git
```

### 3. Cài đặt Supabase
- Tạo tài khoản Supabase tại [supabase.io](https://supabase.io/)
- Tạo project mới và tạo database PostgreSQL
- import file 2 file data từ /final-data/ vào database Supabase
- lấy thông tin kết nối database (DBNAME, DBUSER, DBPASSWORD, DBHOST, DBPORT)


## Hướng dẫn chạy nhanh

### 1. Chạy toàn bộ hệ thống bằng Docker Compose (Khuyên dùng)

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

2. **Chạy lệnh:**
   ```bash
   docker compose up --build
   ```
   - Backend: http://localhost:8000
   - Frontend: http://localhost:3000

> **Lưu ý:** Đảm bảo các file vector (`faiss_index.bin`, `embeddings.pkl`, `chunks.json`) đã có trong `API-chatbot-langgraph/src/services/sec_embeddings/` để chạy được RAG.

---

### 2. Chạy từng thành phần thủ công

#### Backend (API-chatbot-langgraph)
```bash
cd API-chatbot-langgraph
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Tạo file .env như hướng dẫn trên
uvicorn src.main:app --reload
```
- API docs: http://localhost:8000/docs

#### Frontend (Chatbot-FE)
```bash
cd Chatbot-FE
npm install
npm start
```
- FE: http://localhost:3000

---


## Hướng dẫn deploy lên Vercel
1. **Tạo tài khoản Vercel** tại [vercel.com](https://vercel.com/)
2. **Tạo project mới** và chọn repo này
3. **Cấu hình môi trường** trong Vercel:
   - Tạo các biến môi trường tương tự như trong file `.env` ở trên.
   - Đảm bảo các biến môi trường được cấu hình đúng và có giá trị hợp lệ.
4. **Chọn branch để deploy** (thường là `main` hoặc `master`)
5. **Dùng UI để deploy cho dễ**
6. **Chờ Vercel build và deploy** ứng dụng.
7. **Truy cập ứng dụng** qua link Vercel cung cấp. 

## Hướng đẫn forward API từ backend sang frontend bằng NGROK
```bash
ngrok http 8000
```
- Nhớ chạy backend trước khi chạy ngrok (cùng port 8000).
- Sau đó, bạn sẽ nhận được một URL ngrok (ví dụ: `https://abc123.ngrok.io`) mà bạn có thể sử dụng để truy cập API từ frontend hoặc từ bất kỳ đâu trên internet.
```bash
ngrok http --url=reptile-enormous-formerly.ngrok-free.app 8000 --log=stdout
```
- Bạn có thể thay đổi URL trong frontend để gọi API từ ngrok.



## Pipeline xử lý dữ liệu & test/demo

- **final-data/**: Chứa script crawl giá cổ phiếu DJIA, mapping CIK, tạo embedding, chunk dữ liệu, mapping công ty, v.v. Xem `final-data/README.md` để biết chi tiết pipeline dữ liệu tài chính.
- **test-components/**: Chứa notebook demo pipeline LangGraph (`langgraph.ipynb`), test SQL agent (`sql_agent_implementation.py`), test kết nối DB, process PDF, v.v.
- **data/**, **dum-data/**: Chứa dữ liệu test (pdf, csv, tài chính, phương pháp) và dữ liệu mẫu thử nghiệm.

> **Bạn có thể chạy thử các notebook/script trong test-components để hiểu rõ pipeline và kiểm thử các thành phần agent, embedding, SQL, v.v.**

---

## Thông tin chi tiết từng thành phần

- **Backend (API-chatbot-langgraph):** FastAPI, LangChain, LangGraph, FAISS, PostgreSQL, Tavily, OpenAI, Gemini. Xem chi tiết hướng dẫn và cấu trúc trong [`API-chatbot-langgraph/README.md`](./API-chatbot-langgraph/README.md)
- **Frontend (Chatbot-FE):** React, hỗ trợ markdown, highlight code, đăng nhập Google. Xem chi tiết hướng dẫn và cấu trúc trong [`Chatbot-FE/README.md`](./Chatbot-FE/README.md)
- **Pipeline dữ liệu (final-data):** Xem chi tiết pipeline, mapping, embedding, mapping CIK, crawl giá cổ phiếu, v.v. trong [`final-data/README.md`](./final-data/README.md)

---

## Lưu ý & Troubleshooting

- **CORS:** Đảm bảo backend cho phép CORS từ FE.
- **Vector DB:** Nếu thiếu file vector, cần chạy script tạo embedding trước (xem final-data).
- **API Keys:** Đảm bảo các key hợp lệ, còn hạn mức.
- **Database:** Đảm bảo DB Supabase truy cập được từ backend.
- **Dữ liệu:** Nếu muốn mở rộng, hãy thêm/chunk dữ liệu mới và tạo embedding lại.
- **Test/demo:** Có thể chạy các notebook/script trong test-components để kiểm thử pipeline, agent, SQL, v.v.

---