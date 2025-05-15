# Downloader Giá Cổ Phiếu DJIA & Pipeline Dữ Liệu

Script này dùng để tải dữ liệu giá cổ phiếu lịch sử cho tất cả các công ty thuộc chỉ số **Dow Jones Industrial Average (DJIA)**, đồng thời xây dựng pipeline dữ liệu chuẩn cho phân tích hoặc huấn luyện mô hình tài chính.

---

## Tính năng chính

- Tải dữ liệu giá cổ phiếu hàng ngày (Open, High, Low, Close, Adj Close, Volume) cho 30 công ty thuộc DJIA
- Gộp thành một file tổng hợp toàn bộ giá DJIA
- Tạo data embedding cho dữ liệu không cấu trúc
- Tạo bảng dữ liệu cho các công ty thuộc DJIA với các thông tin như mã cổ phiếu, tên công ty, lĩnh vực, ngành nghề, quốc gia, vốn hóa thị trường, tỷ lệ P/E, lợi suất cổ tức, giá cao nhất/thấp nhất trong 52 tuần và mô tả hoạt động kinh doanh.

---

## Cấu trúc cơ sở dữ liệu sinh ra

Hệ thống tạo ra hai bảng liên kết theo quan hệ khóa chính/khóa ngoại:

### 1. `djia_companies.csv` – Thông tin các công ty

| Cột             | Mô tả                         |
|------------------|-------------------------------|
| symbol           | Mã cổ phiếu (khóa chính)      |
| name             | Tên công ty                   |
| sector           | Lĩnh vực kinh doanh           |
| industry         | Ngành cụ thể                  |
| country          | Quốc gia                      |
| website          | Trang web chính thức          |
| market_cap       | Vốn hóa thị trường (BIGINT)   |
| pe_ratio         | Tỷ lệ P/E (FLOAT)             |
| dividend_yield   | Lợi suất cổ tức (%)           |
| 52_week_high     | Giá cao nhất trong 52 tuần    |
| 52_week_low      | Giá thấp nhất trong 52 tuần   |
| description      | Mô tả hoạt động kinh doanh    |

---

### 2. `djia_prices_YYYYMMDD.csv` – Dữ liệu giá cổ phiếu lịch sử

| Cột             | Mô tả                                    |
|------------------|--------------------------------------------|
| Date             | Ngày giao dịch (TIMESTAMP)                |
| Open             | Giá mở cửa                                 |
| High             | Giá cao nhất trong phiên                   |
| Low              | Giá thấp nhất trong phiên                  |
| Close            | Giá đóng cửa                               |
| Adj Close        | Giá điều chỉnh                             |
| Volume           | Khối lượng giao dịch                       |
| Ticker           | Mã cổ phiếu (khóa ngoại tới `djia_companies.symbol`) |

---

## Yêu cầu môi trường

- Python ≥ 3.8
- Các thư viện:
  - `yfinance`
  - `pandas`
  - `requests`


## Hướng dẫn sử dụng

### 1. Sử lý dữ liệu có cấu trúc
```bash
1. Tải xuống mã nguồn từ GitHub hoặc sao chép vào máy tính của bạn.
    # Tải xuống dữ liệu
    python download_djia_companies.py
    python download_djia_stock_prices.py

2. Import dữ liệu vào cơ sở dữ liệu của bạn. ( Sử dụng Supabase ) để import dữ liệu vào cơ sở dữ liệu của bạn. và kéo liên kết giữa 2 bảng `djia_companies` và `djia_prices` theo khóa chính/khóa ngoại. 
````

### 2. Sử lý dữ liệu không cấu trúc
```bash
1. Sử lý làm phẳng dữ liệu không cấu trúc
    # Tải xuống dữ liệu
    python flatten.py

2. Tạo embedding cho dữ liệu không cấu trúc
    # Tạo embedding cho dữ liệu không cấu trúc
    python create_embeddings.py

3. Copy dữ liệu đã tạo embedding vào thư mục `API-chatbot-langgraph/src/services`. để chạy RAG
```


