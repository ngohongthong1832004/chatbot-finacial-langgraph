import psycopg2
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
            {"role": "system", "content": "You are an expert SQL developer specializing in PostgreSQL for financial data analysis. Generate only SQL queries, no explanations."},
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
# Stock Market Database Schema
This database contains comprehensive historical price data for companies in the Dow Jones Industrial Average (DJIA) index from 2023-04-26 to 2025-04-25.

## Tables

### 1. djia_companies
Contains company information and fundamental data.

| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR | Stock ticker symbol (PRIMARY KEY) |
| name | VARCHAR | Full company name |
| sector | VARCHAR | Business sector (e.g., Technology, Healthcare) |
| industry | VARCHAR | Specific industry category |
| country | VARCHAR | Headquarters country |
| website | VARCHAR | Company website URL |
| market_cap | BIGINT | Market capitalization in USD |
| pe_ratio | FLOAT | Price-to-earnings ratio |
| dividend_yield | FLOAT | Annual dividend yield percentage |
| 52_week_high | FLOAT | Highest price in past 52 weeks |
| 52_week_low | FLOAT | Lowest price in past 52 weeks |
| description | TEXT | Company business description |

### 2. djia_prices
Contains daily historical price data (OHLCV).

| Column | Type | Description |
|--------|------|-------------|
| "Date" | TIMESTAMP | Trading date |
| "Open" | FLOAT | Opening price |
| "High" | FLOAT | Highest price during trading day |
| "Low" | FLOAT | Lowest price during trading day |
| "Close" | FLOAT | Closing price |
| "Volume" | BIGINT | Number of shares traded |
| "Dividends" | TEXT/FLOAT | Dividend amount (use ::FLOAT for calculations) |
| "Stock Splits" | TEXT/FLOAT | Stock split ratio |
| "Ticker" | VARCHAR | Stock ticker symbol (FOREIGN KEY to djia_companies.symbol) |

## Critical Database Rules
1. Column names in djia_prices are case-sensitive - ALWAYS use double quotes
2. Always use double quotes for djia_prices columns: "Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "Ticker"
3. Column names in djia_companies do not require quotes
4. Available date range: 2023-04-26 to 2025-04-25 (YYYY-MM-DD)
5. Cast dates when comparing: WHERE p."Date"::date = '2024-03-15'
6. When using aggregate functions with "Dividends", cast to FLOAT: AVG("Dividends"::FLOAT)
7. For percentage calculations: ROUND(((value1 - value2) / value2 * 100)::numeric, 2)

## Available Companies (Ticker - Name)
AAPL - Apple Inc.            | AMGN - Amgen Inc.          | AXP  - American Express
BA   - Boeing Co.            | CAT  - Caterpillar Inc.    | CRM  - Salesforce Inc.
CSCO - Cisco Systems         | CVX  - Chevron Corp.       | DIS  - Walt Disney Co.
DOW  - Dow Inc.              | GS   - Goldman Sachs       | HD   - Home Depot
HON  - Honeywell International| IBM  - IBM                 | INTC - Intel Corp.
JNJ  - Johnson & Johnson     | JPM  - JPMorgan Chase      | KO   - Coca-Cola Co.
MCD  - McDonald's Corp.      | MMM  - 3M Company          | MRK  - Merck & Co.
MSFT - Microsoft Corp.       | NKE  - Nike Inc.           | PG   - Procter & Gamble
TRV  - Travelers Companies   | UNH  - UnitedHealth Group  | V    - Visa Inc.
VZ   - Verizon Communications| WBA  - Walgreens           | WMT  - Walmart Inc.

## Company Sectors
Technology: AAPL, CSCO, IBM, INTC, MSFT, CRM
Healthcare: AMGN, JNJ, MRK, UNH
Financial: AXP, GS, JPM, TRV, V
Industrial: BA, CAT, HON, MMM
Energy: CVX, DOW
Consumer: DIS, HD, KO, MCD, NKE, PG, WMT
Telecommunications: VZ
Retail: WBA

## Common Query Patterns

### Specific Price Lookups
-- Get price on specific date
SELECT p."Date", p."Open", p."High", p."Low", p."Close", p."Volume"
FROM djia_prices p
WHERE p."Ticker" = 'MSFT' 
AND p."Date"::date = '2024-03-15';

-- Get price with company name
SELECT c.name, p."Date", p."Close"
FROM djia_companies c
JOIN djia_prices p ON c.symbol = p."Ticker"
WHERE c.symbol = 'AAPL'
AND p."Date"::date = '2024-03-15';
Date Range Queries
sql-- Get data for specific month
SELECT p."Date", p."Close"
FROM djia_prices p
WHERE p."Ticker" = 'AAPL'
AND p."Date" BETWEEN '2024-03-01' AND '2024-03-31';

-- Q1 2025 date range
AND p."Date" BETWEEN '2025-01-01' AND '2025-03-31';

-- Second half of 2023
AND p."Date" BETWEEN '2023-07-01' AND '2023-12-31';
Statistical Analysis
sql-- Average price over period
SELECT AVG(p."Close") AS avg_close
FROM djia_prices p
WHERE p."Ticker" = 'MSFT'
AND p."Date" BETWEEN '2025-03-01' AND '2025-03-31';

-- Standard deviation
SELECT STDDEV(p."Close")::numeric AS stddev_close
FROM djia_prices p
WHERE p."Ticker" = 'AAPL'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';

-- Count days above threshold
SELECT COUNT(*) AS days_above_threshold
FROM djia_prices p
WHERE p."Ticker" = 'DIS'
AND p."Close" > 90
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';

-- Moving average (30-day)
SELECT p."Date", p."Close",
AVG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date" ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS moving_avg_30
FROM djia_prices p
WHERE p."Ticker" = 'MSFT'
AND p."Date" <= '2024-04-30'
ORDER BY p."Date" DESC
LIMIT 1;
Max/Min Queries
sql-- Highest price in period
SELECT MAX(p."High") AS max_price, p."Date"
FROM djia_prices p
WHERE p."Ticker" = 'AAPL'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';

-- Date of highest close
SELECT p."Date", p."Close"
FROM djia_prices p
WHERE p."Ticker" = 'MSFT'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
ORDER BY p."Close" DESC
LIMIT 1;

-- Lowest closing price with date
SELECT p."Date", p."Close"
FROM djia_prices p
WHERE p."Ticker" = 'VZ'
AND p."Date" BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY p."Close" ASC
LIMIT 1;
Comparative Analysis
sql-- Compare two stocks on same date
SELECT c.name, p."Close"
FROM djia_companies c
JOIN djia_prices p ON c.symbol = p."Ticker"
WHERE p."Date"::date = '2025-01-15'
AND c.symbol IN ('AAPL', 'MSFT')
ORDER BY p."Close" DESC;

-- Find highest price among all companies on date
SELECT c.name, p."Close"
FROM djia_companies c
JOIN djia_prices p ON c.symbol = p."Ticker"
WHERE p."Date"::date = '2024-07-01'
ORDER BY p."Close" DESC
LIMIT 1;

-- Find lowest price among all companies on date
SELECT c.name, p."Close"
FROM djia_companies c
JOIN djia_prices p ON c.symbol = p."Ticker"
WHERE p."Date"::date = '2024-04-01'
ORDER BY p."Close" ASC
LIMIT 1;
Performance Analysis
sql-- Price change percentage between dates
SELECT c.name,
ROUND(((end_prices."Close" - start_prices."Close") / start_prices."Close" * 100)::numeric, 2) AS price_change_percent
FROM djia_companies c
JOIN djia_prices start_prices ON c.symbol = start_prices."Ticker" 
JOIN djia_prices end_prices ON c.symbol = end_prices."Ticker"
WHERE start_prices."Date"::date = '2024-01-02'
AND end_prices."Date"::date = '2024-12-31'
ORDER BY price_change_percent DESC;

-- Largest percentage increase
SELECT c.name,
ROUND(((end_prices."Close" - start_prices."Close") / start_prices."Close" * 100)::numeric, 2) AS price_change_percent
FROM djia_companies c
JOIN djia_prices start_prices ON c.symbol = start_prices."Ticker" 
JOIN djia_prices end_prices ON c.symbol = end_prices."Ticker"
WHERE start_prices."Date"::date = '2024-01-02'
AND end_prices."Date"::date = '2024-12-31'
ORDER BY price_change_percent DESC
LIMIT 1;

-- Absolute price change (dollars)
SELECT c.name, 
(end_prices."Close" - start_prices."Close")::numeric AS price_change_dollars
FROM djia_companies c
JOIN djia_prices start_prices ON c.symbol = start_prices."Ticker" 
JOIN djia_prices end_prices ON c.symbol = end_prices."Ticker"
WHERE start_prices."Date"::date = '2024-01-02'
AND end_prices."Date"::date = '2024-12-31'
AND c.symbol = 'BA';
Dividends and Splits
sql-- Get dividend dates and amounts
SELECT p."Date", p."Dividends"
FROM djia_prices p
WHERE p."Ticker" = 'MSFT'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
AND p."Dividends"::float > 0
ORDER BY p."Date";

-- Count dividends paid
SELECT COUNT(*) AS dividend_count
FROM djia_prices p
WHERE p."Ticker" = 'AAPL'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
AND p."Dividends"::float > 0;

-- Find stock splits
SELECT p."Date", p."Stock Splits"
FROM djia_prices p
WHERE p."Ticker" = 'WMT'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
AND p."Stock Splits"::float > 0;
Trading Volume Analysis
sql-- Highest volume days
SELECT p."Date", p."Ticker", p."Volume"
FROM djia_prices p
WHERE p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
ORDER BY p."Volume" DESC
LIMIT 5;

-- Average daily volume
SELECT AVG(p."Volume") AS avg_volume
FROM djia_prices p
WHERE p."Ticker" = 'AAPL'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';

-- Total volume over period
SELECT SUM(p."Volume") AS total_volume
FROM djia_prices p
WHERE p."Ticker" = 'MSFT'
AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';
Advanced Statistics
sql-- Company with lowest volatility
WITH stock_volatility AS (
  SELECT 
    p."Ticker",
    STDDEV((p."Close" - LAG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date")) / LAG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date")) AS volatility
  FROM 
    djia_prices p
  WHERE 
    p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
  GROUP BY 
    p."Ticker"
)
SELECT 
  c.name, 
  sv.volatility
FROM 
  stock_volatility sv
JOIN 
  djia_companies c ON sv.ticker = c.symbol
ORDER BY 
  sv.volatility ASC
LIMIT 1;

-- Correlations between stocks (simplified)
WITH daily_returns AS (
  SELECT 
    p."Date",
    p."Ticker",
    (p."Close" - LAG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date")) / LAG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date") AS daily_return
  FROM 
    djia_prices p
  WHERE 
    p."Ticker" IN ('AAPL', 'MSFT')
    AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
)
SELECT 
  CORR(a.daily_return, b.daily_return) AS correlation
FROM 
  daily_returns a
JOIN 
  daily_returns b ON a."Date" = b."Date" AND a."Ticker" = 'AAPL' AND b."Ticker" = 'MSFT';
  
--- 
 
Note: To ensure accurate and consistent calculations, here are the standardized formulas and methodologies:

### Return Calculations

1. **Daily Return**:

   $$
   \text{Daily Return}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}
   $$

2. **Cumulative Return**:

   $$
   \text{Cumulative Return} = \frac{\text{Close}_{\text{end}} - \text{Close}_{\text{start}}}{\text{Close}_{\text{start}}}
   $$

3. **Annualized Return**:

   $$
   \text{Annualized Return} = \text{Average Daily Return} \times 252
   $$

4. **Compound Annual Growth Rate (CAGR)**:

   $$
   \text{CAGR} = \left( \frac{\text{Close}_{\text{end}}}{\text{Close}_{\text{start}}} \right)^{\frac{1}{n}} - 1
   $$

   Where $n$ is the number of years.

---

### Volatility Calculations

1. **Daily Volatility**:

   $$
   \text{Daily Volatility} = \text{Standard Deviation of Daily Returns}
   $$

2. **Annualized Volatility**:

   $$
   \text{Annualized Volatility} = \text{Daily Volatility} \times \sqrt{252}
   $$

---

### Risk-Adjusted Metrics

1. **Sharpe Ratio**:

   $$
   \text{Sharpe Ratio} = \frac{\text{Annualized Return} - \text{Risk-Free Rate}}{\text{Annualized Volatility}}
   $$

2. **Beta** (Relative to a Benchmark):

   $$
   \beta = \frac{\text{Covariance}(\text{Asset Returns}, \text{Benchmark Returns})}{\text{Variance}(\text{Benchmark Returns})}
   $$

3. **Correlation**:

   $$
   \text{Correlation} = \frac{\text{Covariance}(\text{Asset Returns}, \text{Benchmark Returns})}{\text{Standard Deviation of Asset Returns} \times \text{Standard Deviation of Benchmark Returns}}
   $$

---

### Date-Specific Metrics

1. **Moving Average (e.g., 30-day)**:

   $$
   \text{Moving Average}_t = \frac{1}{30} \sum_{i=0}^{29} \text{Close}_{t-i}
   $$

2. **Standard Deviation of Closing Prices**:

   $$
   \text{Standard Deviation} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\text{Close}_i - \text{Mean Close})^2}
   $$

---

### Dividend Metrics

1. **Dividend Yield**:

   $$
   \text{Dividend Yield} = \frac{\text{Annual Dividends per Share}}{\text{Price per Share}}
   $$

2. **Total Dividends Paid**:

   $$
   \text{Total Dividends} = \text{Dividend per Share} \times \text{Number of Shares}
   $$

---

### Volume Metrics

1. **Average Daily Trading Volume**:

   $$
   \text{Average Volume} = \frac{\sum \text{Daily Volume}}{\text{Number of Trading Days}}
   $$

2. **Total Trading Volume**:

   $$
   \text{Total Volume} = \sum \text{Daily Volume}
   $$

---

### Comparative Metrics

1. **Percentage Change Over Period**:

   $$
   \text{Percentage Change} = \left( \frac{\text{Close}_{\text{end}} - \text{Close}_{\text{start}}}{\text{Close}_{\text{start}}} \right) \times 100\%
   $$

2. **Absolute Change Over Period**:

   $$
   \text{Absolute Change} = \text{Close}_{\text{end}} - \text{Close}_{\text{start}}
   $$

---

### Maximum Drawdown

1. **Maximum Drawdown**:

   $$
   \text{Max Drawdown} = \max \left( \frac{\text{Peak} - \text{Trough}}{\text{Peak}} \right)
   $$

   Where "Peak" is the highest value before a decline, and "Trough" is the lowest value after the peak.

---

### Other Metrics

1. **Median Closing Price**:

   $$
   \text{Median} = \text{Middle value of sorted closing prices}
   $$

2. **Standard Deviation of Daily Returns**:

   $$
   \text{Standard Deviation} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\text{Daily Return}_i - \text{Mean Daily Return})^2}
   $$

  
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


    # 2. Tạo sample_data (mô phỏng hoặc tải từ metadata)
    sample_data = get_schema_and_samples(conn=None)  # đã chứa key 'metadata_text'
    sample_description = sample_data.get("metadata_text", "")

    # 3. Prompt kết hợp cả schema và sample
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert SQL developer specialized in financial data queries for the stock market.

# Task
Generate a PostgreSQL query that accurately answers the user's question about stock market data.

# Database Schema
{schema}

# User Question
{question}

# Instructions
1. IMPORTANT: Return ONLY the SQL query with no explanations or markdown formatting
2. Follow PostgreSQL syntax exactly
3. Remember to use double quotes for column names in the djia_prices table
4. Only query data within the date range: 2023-04-26 to 2025-04-25
5. For division or percentage calculations, cast to numeric or float to avoid integer division
6. Use meaningful table aliases (e.g., 'c' for companies, 'p' for prices)
7. For date-specific queries, use "Date"::date format when comparing with date literals
8. Include LIMIT clauses for queries that could return many rows
9. NEVER put quotes around expressions or comparisons. Only quote column names like "Ticker" or "Close", NOT expressions like a."Ticker" <> b."Ticker". Write them as a."Ticker" <> b."Ticker" (correct), not a."Ticker <> b."Ticker" (WRONG).
10. For comparative or ranking questions (e.g., compare multiple companies, find highest/lowest/most volatile), return **2 or 10 relevant rows**, do NOT limit with `LIMIT 1` or filters. The system will handle ranking or filtering later.
11. NEVER use window functions (e.g., LAG, LEAD, ROW_NUMBER) inside WHERE clauses. Instead, use a CTE or subquery to compute them first, then filter or sort from the outer query.
""")

    chain = prompt | RunnableLambda(call_openrouter) | StrOutputParser()


    try:
        sql_query = chain.invoke({
            "schema": sample_description,
            "question": question
        })
        return sql_query.strip()
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        return ""
