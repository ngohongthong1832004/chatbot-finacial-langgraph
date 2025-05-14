# Downloader Giá Cổ Phiếu DJIA & Pipeline Dữ Liệu

Script này dùng để tải dữ liệu giá cổ phiếu lịch sử cho tất cả các công ty thuộc chỉ số Dow Jones Industrial Average (DJIA) từ 01/01/2022 đến ngày hiện tại.

## Tính năng

- Tải dữ liệu giá cổ phiếu hàng ngày (Open, High, Low, Close, Adj Close, Volume) cho 30 công ty DJIA
- Tự động retry khi bị giới hạn API
- Lưu file CSV riêng cho từng công ty
- Tạo file CSV tổng hợp cho toàn bộ DJIA
- Hiển thị tiến trình và tổng kết chi tiết

## Yêu cầu

- Python 3.7 trở lên
- Thư viện: yfinance, pandas, requests

## Cài đặt

1. Clone hoặc tải repo này
2. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## Sử dụng

Chạy script bằng Python:

```bash
python download_djia_stock_prices.py
```

## Kết quả đầu ra

Script tạo thư mục `stock_prices` chứa:

1. File CSV riêng cho từng công ty (vd: `AAPL_prices.csv`)
2. File tổng hợp toàn bộ DJIA (vd: `djia_prices_20240501.csv`)

Mỗi file gồm các cột:
- Date: Ngày giao dịch
- Open: Giá mở cửa
- High: Giá cao nhất
- Low: Giá thấp nhất
- Close: Giá đóng cửa
- Adj Close: Giá điều chỉnh
- Volume: Khối lượng giao dịch
- Ticker: Mã cổ phiếu

## Lưu ý

- Script sử dụng Yahoo Finance API qua package yfinance
- Mặc định lấy dữ liệu từ 01/01/2022 đến hiện tại
- Muốn đổi khoảng thời gian, sửa biến `start_date` và `end_date` trong hàm `main()`

## Mapping CIK của SEC

Repo này có mapping giữa mã CIK (Central Index Key) của SEC và các công ty DJIA, giúp liên kết báo cáo tài chính SEC với từng công ty.

### CIK là gì?

CIK là mã định danh duy nhất do SEC cấp cho các công ty/cá nhân nộp báo cáo tài chính. Một công ty có thể có nhiều CIK do sáp nhập, mua lại...

### File mapping

- **cik_to_company_mapping.csv** – Bảng mapping chi tiết (CIK, tên công ty, ticker, ngành, phương pháp mapping...)
- **cik_to_company_mapping.json** – Mapping dạng JSON
- **cik_lookup.json** – Lookup đơn giản cho tra cứu nhanh

### Ví dụ tra cứu CIK bằng Python

```python
import json

with open('references/cik_lookup.json', 'r') as f:
    cik_lookup = json.load(f)

cik = '0000320193'  # Apple Inc.
if cik in cik_lookup:
    company_info = cik_lookup[cik]
    print(f"CIK {cik} thuộc về {company_info['name']} ({company_info['ticker']})")
else:
    print(f"Không tìm thấy CIK {cik} trong mapping")
```

### Sinh mapping

Mapping được tạo bằng script `map_cik_to_companies_fixed.py`:
1. Quét toàn bộ báo cáo tài chính DJIA
2. Trích xuất CIK
3. Mapping tự động + thủ công
4. Sinh file mapping CSV/JSON