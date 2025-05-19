import psycopg2
import pandas as pd

import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import requests

# Load biến môi trường từ file .env
load_dotenv()

def call_openrouter(prompt_obj) -> str:
    prompt = prompt_obj.to_string()  

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert grader assessing relevance of a retrieved document to a user question. Answer only 'yes' or 'no'."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"OpenRouter API error: {res.status_code} - {res.text}")

# Dùng os.environ để truy cập
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def get_db_config():
    return {
    "dbname": os.getenv("DBNAME"),
    "user": os.getenv("DBUSER"),
    "password": os.getenv("DBPASSWORD"),
    "host": os.getenv("DBHOST"),
    "port": int(os.getenv("DBPORT", 5432)),
    "sslmode": os.getenv("SSL_MODE", "require")
}

def connect_to_database():
    try:
        conn = psycopg2.connect(**get_db_config())
        print("✅ Successfully connected to PostgreSQL.")
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def execute_sql_query(conn, query):
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"❌ SQL execution error: {e}")
        return None

def get_schema_and_samples(conn=None):
    # Luôn dùng metadata dạng text cung cấp thủ công
    try:
        # Đây là metadata bạn yêu cầu dùng để thay cho việc truy vấn schema
        metadata_text = """
            Cơ sở dữ liệu này lưu trữ thông tin về các công ty trong chỉ số Dow Jones Industrial Average (DJIA) và dữ liệu giá cổ phiếu lịch sử của họ.
            Đây là hai bảng chính trong cơ sở dữ liệu mà hệ thống có tham khảo nó để tạo câu truy vấn SQL:
            CREATE TABLE public.djia_companies (
                symbol text NOT NULL,
                name text,
                sector text,
                industry text,
                country text,
                website text,
                market_cap bigint,
                pe_ratio double precision,
                dividend_yield double precision,
                "52_week_high" double precision,
                "52_week_low" double precision,
                description text
            );

            ALTER TABLE public.djia_companies OWNER TO postgres;

            CREATE TABLE public.djia_prices (
                "Date" timestamp with time zone,
                "Open" double precision,
                "High" double precision,
                "Low" double precision,
                "Close" double precision,
                "Volume" bigint,
                "Dividends" text,
                "Stock Splits" text,
                "Ticker" text NOT NULL
            );
            ALTER TABLE public.djia_prices OWNER TO postgres;
            
            Các bảng:

            djia_companies: Thông tin về các công ty DJIA.

            symbol (VARCHAR): Mã cổ phiếu (Khóa chính).
            name (VARCHAR): Tên công ty.
            sector (VARCHAR): Lĩnh vực kinh doanh.
            industry (VARCHAR): Ngành cụ thể.
            country (VARCHAR): Quốc gia trụ sở.
            Các cột khác: website, market_cap (BIGINT), pe_ratio (FLOAT), dividend_yield (FLOAT), 52_week_high (FLOAT), 52_week_low (FLOAT), description (TEXT).

            djia_prices: Dữ liệu giá cổ phiếu lịch sử hàng ngày.

            "Date" (TIMESTAMP): Ngày giao dịch.
            "Open" (FLOAT): Giá mở cửa.
            "High" (FLOAT): Giá cao nhất.
            "Low" (FLOAT): Giá thấp nhất.
            "Close" (FLOAT): Giá đóng cửa.
            "Volume" (INTEGER): Khối lượng giao dịch.
            "Dividends" (FLOAT hoặc TEXT, nên ép kiểu FLOAT khi dùng với AVG/SUM): Cổ tức.
            "Stock Splits" (FLOAT): Tỷ lệ chia tách cổ phiếu.
            "Ticker" (VARCHAR): Mã cổ phiếu (Khóa ngoại tham chiếu djia_companies.symbol).

            Tài liệu hệ thống một số mã trong cột Ticker của bảng djia_prices:
            AAPL - Apple Inc.
            AMGN - Amgen Inc.
            AXP  - American Express
            BA   - Boeing Co.
            CAT  - Caterpillar Inc.
            CRM  - Salesforce Inc.
            CSCO - Cisco Systems
            CVX  - Chevron Corp.
            DIS  - Walt Disney Co.
            DOW  - Dow Inc.
            GS   - Goldman Sachs
            HD   - Home Depot
            HON  - Honeywell International
            IBM  - International Business Machines
            INTC - Intel Corp.
            JNJ  - Johnson & Johnson
            JPM  - JPMorgan Chase
            KO   - Coca-Cola Co.
            MCD  - McDonald's Corp.
            MMM  - 3M Company
            MRK  - Merck & Co.
            MSFT - Microsoft Corp.
            NKE  - Nike Inc.
            PG   - Procter & Gamble
            TRV  - Travelers Companies
            UNH  - UnitedHealth Group
            V    - Visa Inc.
            VZ   - Verizon Communications
            WBA  - Walgreens Boots Alliance
            WMT  - Walmart Inc.

            Lưu ý quan trọng cho PostgreSQL:

            Tên các cột trong bảng djia_prices là phân biệt chữ hoa/thường.
            Luôn sử dụng dấu ngoặc kép cho các cột này trong truy vấn: "Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Ticker".
            Tên các cột trong bảng djia_companies không cần dấu ngoặc kép.

            Ví dụ truy vấn SQL:

            1. Lấy 10 công ty có P/E ratio cao nhất:
            SELECT name, pe_ratio
            FROM djia_companies
            ORDER BY pe_ratio DESC
            LIMIT 10;

            2. Lấy giá đóng cửa cao nhất của mỗi công ty trong ngày gần nhất:
            SELECT c.name, MAX(p."Close") AS max_close
            FROM djia_companies c
            JOIN djia_prices p ON c.symbol = p."Ticker"
            WHERE p."Date" = (SELECT MAX("Date") FROM djia_prices)
            GROUP BY c.name;

            3. Lấy giá đóng cửa của Apple trong tháng 3 năm 2024:
            SELECT c.name, p."Date", p."Close"
            FROM djia_companies c
            JOIN djia_prices p ON c.symbol = p."Ticker"
            WHERE p."Date" = (
                SELECT MAX("Date") FROM djia_prices p2 WHERE p2."Ticker" = p."Ticker"
            )
            ORDER BY p."Close" DESC;

            4. Lấy giá đóng cửa trung bình của Apple trong tháng 3 năm 2024:
            SELECT AVG("Close") AS avg_close
            FROM djia_prices
            WHERE "Ticker" = 'AAPL'
            AND "Date" BETWEEN '2024-03-01' AND '2024-03-31';

            5. Lấy danh sách các công ty trong lĩnh vực công nghệ:
            SELECT name, sector, industry, market_cap
            FROM djia_companies
            WHERE sector = 'Technology';

            6. Lấy danh sách các công ty có cổ tức cao nhất:
            SELECT c.name, p."Open", p."Close"
            FROM djia_companies c
            JOIN djia_prices p ON c.symbol = p."Ticker"
            WHERE DATE_TRUNC('day', p."Date") = '2024-05-01'::DATE

            7. Lấy danh sách 5 công ty có cổ tức trung bình cao nhất:
            SELECT c.name, AVG(p."Dividends"::FLOAT) AS avg_dividend
            FROM djia_companies c
            JOIN djia_prices p ON c.symbol = p."Ticker"
            GROUP BY c.name
            ORDER BY avg_dividend DESC
            LIMIT 5;

            8. Lấy giá đóng cửa của Microsoft vào ngày 2024-03-15:
            SELECT "Date", "Ticker", "Close"
            FROM djia_prices
            WHERE "Ticker" = 'MSFT'
            AND "Date"::date = '2024-03-15';

        """
        return {"metadata_text": metadata_text.strip()}
    except Exception as e:
        print(f"❌ Error loading text metadata: {e}")
        return {"error": str(e)}

def load_metadata_from_txt(file_path: str) -> str:
    try:
        abs_path = os.path.abspath(file_path)
        print(f"📄 Attempting to load metadata from: {abs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return ""    

def generate_sql_query(question, schema_info=None):
    """Generate SQL query from natural language question using LLM"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json

    # print(f"Schema preview:\n{json.dumps(schema_info, indent=2)}")
    # print(f"Prompted question:\n{question}")

    # 1. Tạo biến schema_description
    if schema_info and "metadata_text" in schema_info:
        schema_description = schema_info["metadata_text"]
    else:
        schema_description = "No schema information available."

    # 2. Tạo sample_data (mô phỏng hoặc tải từ metadata)
    sample_data = get_schema_and_samples(conn=None)  # đã chứa key 'metadata_text'
    sample_description = sample_data.get("metadata_text", "")

    # 3. Prompt kết hợp cả schema và sample
    prompt = ChatPromptTemplate.from_template("""
You are an expert SQL developer. Generate a SQL query to answer the user's question.
Use the following database schema information:

{schema}

And some reference sample or metadata (if needed):

{sample_data}

User's question: {question}

Return ONLY the SQL query without any explanation or markdown formatting.
Make sure the query is correct PostgreSQL syntax.
""")

    chain = prompt | RunnableLambda(call_openrouter) | StrOutputParser()

    try:
        sql_query = chain.invoke({
            "schema": schema_description,
            "sample_data": sample_description,
            "question": question
        })
        return sql_query.strip()
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        return ""
