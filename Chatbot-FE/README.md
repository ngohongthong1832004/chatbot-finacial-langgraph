# Chatbot Frontend (React)

Giao diện người dùng cho hệ thống Chatbot Agentic RAG, xây dựng bằng React.

## Tính năng

- Giao diện chat hiện đại, hỗ trợ markdown và highlight code
- Kết nối API backend để hỏi đáp thông minh
- Hỗ trợ đăng nhập Google (nếu backend bật OAuth)

## Khởi động nhanh

### 1. Chạy bằng Docker (Khuyên dùng)

> **Yêu cầu:** Đã cài Docker và docker-compose, backend đã cấu hình đúng CORS.

```bash
docker compose up --build
```
- Frontend sẽ chạy tại: [http://localhost:3000](http://localhost:3000)

**Hoặc chỉ build FE:**
```bash
cd Chatbot-FE
docker build -t chatbot-fe .
docker run -p 3000:3000 chatbot-fe
```

### 2. Chạy thủ công với Node.js

> **Yêu cầu:** Node.js >= 18, npm

```bash
cd Chatbot-FE
npm install
npm start
```
- Ứng dụng sẽ chạy tại: [http://localhost:3000](http://localhost:3000)

## Cấu hình

- Để đổi địa chỉ backend API, sửa biến trong file `.env` hoặc trực tiếp trong code (thường là `src/App.js`).
- Nếu dùng OAuth, đảm bảo backend đã cấu hình đúng Google OAuth và CORS.

## Cấu trúc thư mục

```
Chatbot-FE/
├── public/
│   ├── index.html         # HTML gốc
│   ├── favicon.ico        # Icon
│   ├── logo192.png        # Logo
│   ├── logo512.png        # Logo lớn
│   ├── manifest.json      # Cấu hình PWA
│   └── robots.txt
├── src/
│   ├── App.js             # Thành phần chính của app
│   ├── App.css            # CSS chính
│   ├── index.js           # Điểm vào React
│   ├── index.css          # CSS gốc
│   └── reportWebVitals.js # Đo hiệu năng
├── package.json           # Thông tin và dependency
├── package-lock.json      # Khóa dependency
├── Dockerfile             # Dockerfile FE
└── .gitignore
```

## Scripts npm

- `npm start` – Chạy server phát triển
- `npm run build` – Build bản production

## Lưu ý

- Đảm bảo backend đã chạy và cho phép CORS từ frontend.
- Nếu đổi port hoặc domain backend, cần sửa lại endpoint API trong FE.
- Nếu gặp lỗi CORS, kiểm tra lại cấu hình backend.

---

**Bạn có thể bổ sung thêm hướng dẫn tùy chỉnh giao diện, hoặc ví dụ cấu hình .env nếu muốn!** 