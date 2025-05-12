import psycopg2
import pandas as pd

import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()


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
            "Dividends" (FLOAT): Cổ tức.
            "Stock Splits" (FLOAT): Tỷ lệ chia tách cổ phiếu.
            "Ticker" (VARCHAR): Mã cổ phiếu (Khóa ngoại tham chiếu djia_companies.symbol).

            Quan hệ:

            djia_prices."Ticker" tham chiếu djia_companies.symbol (Quan hệ Một-Nhiều: Một công ty có nhiều dòng giá).

            Lưu ý quan trọng cho PostgreSQL:

            Tên các cột trong bảng djia_prices là phân biệt chữ hoa/thường.
            Luôn sử dụng dấu ngoặc kép cho các cột này trong truy vấn: "Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Ticker".
            Tên các cột trong bảng djia_companies không cần dấu ngoặc kép.
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

    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)

    # Nếu có metadata text => dùng luôn
    if schema_info and "metadata_text" in schema_info:
        schema_description = schema_info["metadata_text"]
    elif schema_info:
        schema_parts = []
        for table_name, table_info in schema_info.items():
            columns = ", ".join([f"{col['column_name']} ({col['data_type']})"
                                 for col in table_info['columns']])
            schema_parts.append(f"Table: {table_name}\nColumns: {columns}")

            if table_info['sample_data']:
                sample = str(table_info['sample_data'][0])
                schema_parts.append(f"Sample row: {sample}")

        schema_description = "\n\n".join(schema_parts)
    else:
        schema_description = "Unknown schema. Try to generate a generic SQL query."

    prompt = ChatPromptTemplate.from_template("""
You are an expert SQL developer. Generate a SQL query to answer the user's question.
Use the following database schema information:

{schema}

User's question: {question}

Return ONLY the SQL query without any explanation or markdown formatting.
Make sure the query is correct PostgreSQL syntax.
""")
    
    chain = prompt | llm | StrOutputParser()

    try:
        sql_query = chain.invoke({
            "schema": schema_description,
            "question": question
        })
        return sql_query.strip()
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        return ""
