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
      metadata_text = """
# Stock Market Database Schema - Optimized

## Database Overview
- **Dataset**: Dow Jones Industrial Average (DJIA) companies
- **Date Range**: 2023-04-26 to 2025-04-25
- **Tables**: 2 (companies + daily prices)
- **Companies**: 30 DJIA constituents

---

## Table Structure

### djia_companies (Company Master Data)
```sql
symbol VARCHAR PRIMARY KEY    -- Stock ticker (e.g., 'AAPL')
name VARCHAR                  -- Company name (e.g., 'Apple Inc.')
sector VARCHAR               -- Business sector
industry VARCHAR             -- Industry category
country VARCHAR              -- HQ location
website VARCHAR              -- Company URL
market_cap BIGINT           -- Market cap (USD)
pe_ratio FLOAT              -- P/E ratio
dividend_yield FLOAT        -- Dividend yield %
52_week_high FLOAT          -- 52-week high price
52_week_low FLOAT           -- 52-week low price
description TEXT            -- Business description
```

### djia_prices (Daily OHLCV Data)
```sql
"Date" TIMESTAMP            -- Trading date (QUOTED COLUMN)
"Open" FLOAT               -- Opening price (QUOTED)
"High" FLOAT               -- Day high (QUOTED)
"Low" FLOAT                -- Day low (QUOTED)
"Close" FLOAT              -- Closing price (QUOTED)
"Volume" BIGINT            -- Shares traded (QUOTED)
"Dividends" TEXT/FLOAT     -- Dividend amount (QUOTED)
"Stock Splits" TEXT/FLOAT  -- Split ratio (QUOTED)
"Ticker" VARCHAR           -- Stock symbol FK (QUOTED)
```

---

## Critical Query Rules

### Column Naming
- **djia_prices**: ALL columns MUST use double quotes: `"Date"`, `"Close"`, etc.
- **djia_companies**: No quotes needed: `symbol`, `name`, etc.

### Date Handling
```sql
-- Cast dates for comparison
WHERE p."Date"::date = '2024-03-15'

-- Date ranges
WHERE p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
```

### Type Casting
```sql
-- Dividends calculations
AVG("Dividends"::FLOAT)

-- Percentage calculations
ROUND(((new_val - old_val) / old_val * 100)::numeric, 2)
```

---

## Company Reference

### By Sector
| Sector | Tickers |
|--------|---------|
| **Basic Materials** | DOW |
| **Communication Services** | DIS, VZ |
| **Consumer Cyclical** | HD, MCD, NKE |
| **Consumer Defensive** | KO, PG, WMT |
| **Energy** | CVX |
| **Financial Services** | AXP, GS, JPM, TRV, V |
| **Healthcare** | AMGN, JNJ, MRK, UNH, WBA |
| **Industrials** | BA, CAT, HON, MMM |
| **Technology** | AAPL, CRM, CSCO, IBM, INTC, MSFT |

### Complete Company List
```
AAPL - Apple Inc.                        | AMGN - Amgen Inc.
AXP  - American Express Company          | BA   - Boeing Company (The)
CAT  - Caterpillar, Inc.                 | CRM  - Salesforce, Inc.
CSCO - Cisco Systems, Inc.               | CVX  - Chevron Corporation
DIS  - Walt Disney Company (The)         | DOW  - Dow Inc.
GS   - Goldman Sachs Group, Inc. (The)   | HD   - Home Depot, Inc. (The)
HON  - Honeywell International Inc.      | IBM  - International Business Machines
INTC - Intel Corporation                 | JNJ  - Johnson & Johnson
JPM  - JP Morgan Chase & Co.             | KO   - Coca-Cola Company (The)
MCD  - McDonald's Corporation            | MMM  - 3M Company
MRK  - Merck & Company, Inc.             | MSFT - Microsoft Corporation
NKE  - Nike, Inc.                        | PG   - Procter & Gamble Company (The)
TRV  - The Travelers Companies, Inc.     | UNH  - UnitedHealth Group Incorporated
V    - Visa Inc.                         | VZ   - Verizon Communications Inc.
WBA  - Walgreens Boots Alliance, Inc.    | WMT  - Walmart Inc.
```

---

## Common Query Templates

### Price Lookups
```sql
-- Single stock, specific date
SELECT "Date", "Close" FROM djia_prices 
WHERE "Ticker" = 'AAPL' AND "Date"::date = '2024-03-15';

-- With company name
SELECT c.name, p."Close" FROM djia_companies c
JOIN djia_prices p ON c.symbol = p."Ticker"
WHERE c.symbol = 'MSFT' AND p."Date"::date = '2024-03-15';
```

### Time Series Analysis
```sql
-- Price range (month)
WHERE "Date" BETWEEN '2024-03-01' AND '2024-03-31'

-- Quarter ranges
Q1: '2024-01-01' AND '2024-03-31'
Q2: '2024-04-01' AND '2024-06-30'
Q3: '2024-07-01' AND '2024-09-30'
Q4: '2024-10-01' AND '2024-12-31'

-- Year-over-year
WHERE "Date" BETWEEN '2023-01-01' AND '2024-12-31'
```

### Statistical Queries
```sql
-- Basic stats
SELECT AVG("Close"), MIN("Close"), MAX("Close"), STDDEV("Close")
FROM djia_prices WHERE "Ticker" = 'AAPL';

-- Moving average (30-day)
SELECT "Date", "Close",
  AVG("Close") OVER (ORDER BY "Date" ROWS 29 PRECEDING) as ma_30
FROM djia_prices WHERE "Ticker" = 'MSFT';

-- Performance calculation
ROUND(((end_price - start_price) / start_price * 100)::numeric, 2) as pct_change
```

### Comparative Analysis
```sql
-- Top performers
SELECT c.name, (end_p."Close" - start_p."Close") / start_p."Close" * 100 as return_pct
FROM djia_companies c
JOIN djia_prices start_p ON c.symbol = start_p."Ticker"
JOIN djia_prices end_p ON c.symbol = end_p."Ticker"
WHERE start_p."Date"::date = '2024-01-01'
  AND end_p."Date"::date = '2024-12-31'
ORDER BY return_pct DESC;
```

---

## Financial Calculation Standards

### Returns & Performance
| Metric | Formula | SQL Example |
|--------|---------|-------------|
| **Daily Return** | `(Close_t - Close_t-1) / Close_t-1` | `LAG("Close") OVER (ORDER BY "Date")` |
| **Period Return** | `(Close_end - Close_start) / Close_start` | `((end_p."Close" - start_p."Close") / start_p."Close")` |
| **Cumulative Return** | Same as Period Return | `ROUND((...) * 100, 2)` for percentage |
| **Total Return** | `(Final_Price + Dividends - Initial_Price) / Initial_Price` | Include dividend adjustments |
| **CAGR** | `(Close_end/Close_start)^(1/years) - 1` | `POWER(ratio, 1.0/years) - 1` |
| **Annualized Return** | `Daily_Return_Avg × 252` | Assumes 252 trading days |

### Risk & Volatility
| Metric | Formula | SQL Implementation |
|--------|---------|-------------------|
| **Daily Volatility** | `STDDEV(daily_returns)` | `STDDEV((p."Close" - LAG(p."Close"))...)` |
| **Annualized Vol** | `Daily_Vol × √252` | `daily_vol * SQRT(252)` |
| **Standard Dev Price** | `STDDEV(closing_prices)` | `STDDEV(p."Close")` |
| **Max Drawdown** | `MAX((Peak - Trough) / Peak)` | Requires window functions |
| **Beta** | `COV(stock, market) / VAR(market)` | Correlation-based calculation |

### Advanced Metrics
| Metric | Formula | Notes |
|--------|---------|-------|
| **Sharpe Ratio** | `(Ann_Return - Risk_Free_Rate) / Ann_Vol` | Assume 4% risk-free rate |
| **Correlation** | `CORR(returns_A, returns_B)` | Between two stocks |
| **Median Price** | `PERCENTILE_CONT(0.5)` | Middle value of price series |
| **Moving Average** | `AVG() OVER (ROWS n PRECEDING)` | n-day rolling average |
| **Dividend Yield** | `(Annual_Dividends / Current_Price) * 100` | As percentage |

### Volume & Trading
| Metric | Calculation | Purpose |
|--------|-------------|---------|
| **Avg Daily Volume** | `AVG("Volume")` | Liquidity measure |
| **Total Volume** | `SUM("Volume")` | Period trading activity |
| **Volume-Weighted Price** | `SUM(Price × Volume) / SUM(Volume)` | VWAP calculation |
| **High/Low Analysis** | `MAX("High"), MIN("Low")` | Price extremes |

### Threshold Analysis
| Query Type | SQL Pattern | Example |
|------------|-------------|---------|
| **Days Above Price** | `COUNT(*) WHERE "Close" > threshold` | `WHERE "Close" > 200` |
| **% Days Above** | `(COUNT(*) WHERE condition) / COUNT(*)` | Trading days percentage |
| **Within 1 Std Dev** | `WHERE ABS("Close" - avg) <= stddev` | Normal distribution analysis |
| **Price Ranking** | `RANK() OVER (ORDER BY "Close")` | Relative positioning |

---

## Advanced Query Patterns

### Complex Statistical Queries
```sql
-- Standard deviation of closing prices
SELECT STDDEV(p."Close")::numeric AS price_stddev
FROM djia_prices p
WHERE p."Ticker" = 'AAPL' AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';

-- Days within 1 standard deviation
WITH stats AS (
  SELECT AVG(p."Close") as avg_price, STDDEV(p."Close") as std_price
  FROM djia_prices p WHERE p."Ticker" = 'BA' AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
)
SELECT COUNT(*) as days_within_1std
FROM djia_prices p, stats s
WHERE p."Ticker" = 'BA' 
  AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
  AND ABS(p."Close" - s.avg_price) <= s.std_price;

-- Median closing price
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY p."Close") as median_price
FROM djia_prices p
WHERE p."Ticker" = 'DIS' AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31';
```

### Correlation Analysis
```sql
-- Correlation between two stocks
WITH daily_returns AS (
  SELECT p."Date", p."Ticker",
    (p."Close" - LAG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date")) 
    / LAG(p."Close") OVER (PARTITION BY p."Ticker" ORDER BY p."Date") AS daily_return
  FROM djia_prices p
  WHERE p."Ticker" IN ('AAPL', 'MSFT') 
    AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
)
SELECT CORR(aapl.daily_return, msft.daily_return) as correlation
FROM daily_returns aapl
JOIN daily_returns msft ON aapl."Date" = msft."Date"
WHERE aapl."Ticker" = 'AAPL' AND msft."Ticker" = 'MSFT';
```

### Performance Rankings
```sql
-- Top/Bottom performers by total return
WITH performance AS (
  SELECT c.name,
    ROUND(((end_p."Close" - start_p."Close") / start_p."Close" * 100)::numeric, 2) as total_return
  FROM djia_companies c
  JOIN djia_prices start_p ON c.symbol = start_p."Ticker" 
  JOIN djia_prices end_p ON c.symbol = end_p."Ticker"
  WHERE start_p."Date"::date = '2024-01-02'
    AND end_p."Date"::date = '2024-12-31'
)
SELECT name, total_return,
  RANK() OVER (ORDER BY total_return DESC) as rank_best,
  RANK() OVER (ORDER BY total_return ASC) as rank_worst
FROM performance
ORDER BY total_return DESC;
```

### Volatility & Risk Analysis
```sql
-- Annualized volatility
WITH daily_returns AS (
  SELECT (p."Close" - LAG(p."Close") OVER (ORDER BY p."Date")) 
         / LAG(p."Close") OVER (ORDER BY p."Date") AS daily_return
  FROM djia_prices p
  WHERE p."Ticker" = 'AAPL' AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
)
SELECT STDDEV(daily_return) * SQRT(252) * 100 as annualized_volatility_pct
FROM daily_returns
WHERE daily_return IS NOT NULL;

-- Maximum drawdown calculation
WITH price_peaks AS (
  SELECT p."Date", p."Close",
    MAX(p."Close") OVER (ORDER BY p."Date" ROWS UNBOUNDED PRECEDING) as running_max
  FROM djia_prices p
  WHERE p."Ticker" = 'MSFT' AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
)
SELECT MAX((running_max - "Close") / running_max * 100) as max_drawdown_pct
FROM price_peaks;
```

### Dividend Analysis
```sql
-- All dividend payments with dates
SELECT p."Date", p."Dividends"::float as dividend_amount
FROM djia_prices p
WHERE p."Ticker" = 'MSFT' 
  AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
  AND p."Dividends"::float > 0
ORDER BY p."Date";

-- Annual dividend yield calculation
WITH dividends AS (
  SELECT SUM(p."Dividends"::float) as annual_dividends
  FROM djia_prices p
  WHERE p."Ticker" = 'UNH' AND p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
    AND p."Dividends"::float > 0
),
current_price AS (
  SELECT p."Close" FROM djia_prices p
  WHERE p."Ticker" = 'UNH' AND p."Date"::date = '2024-12-31'
)
SELECT (d.annual_dividends / cp."Close" * 100) as dividend_yield_pct
FROM dividends d, current_price cp;
```

### Volume Analysis
```sql
-- Highest volume days
SELECT p."Date", p."Volume", c.name
FROM djia_prices p
JOIN djia_companies c ON p."Ticker" = c.symbol
WHERE p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
ORDER BY p."Volume" DESC
LIMIT 5;

-- Average volume comparison
SELECT c.name, AVG(p."Volume") as avg_volume,
  RANK() OVER (ORDER BY AVG(p."Volume") DESC) as volume_rank
FROM djia_companies c
JOIN djia_prices p ON c.symbol = p."Ticker"
WHERE p."Date" BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY c.name
ORDER BY avg_volume DESC;
```

### Query Optimization
1. **Always quote djia_prices columns**: `"Date"`, `"Close"`, `"Volume"`
2. **Cast dates properly**: `"Date"::date = '2024-01-01'`
3. **Use proper JOINs**: Link tables via `symbol = "Ticker"`
4. **Index-friendly ranges**: Use BETWEEN for date ranges

### Common Pitfalls to Avoid
- ❌ Unquoted djia_prices columns
- ❌ Wrong date format (use YYYY-MM-DD)
- ❌ Missing type casts for calculations
- ❌ Comparing dates without ::date cast

### Performance Tips
- Use specific date ranges to limit data
- Index on "Ticker" and "Date" for faster queries
- LIMIT results for large datasets
- Use EXPLAIN PLAN for complex queries      
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
4. Do not assume NULLs in data. All values in the database are present. If a metric or event did not occur (e.g., no dividend, no stock split), its value will be 0.0 — not NULL. Therefore, use conditions like "Dividends"::float > 0 or "Stock Splits"::float > 1 instead of checking for IS NOT NULL.
5. For all arithmetic calculations (especially division or percentage), explicitly cast both operands to numeric or float before the operation. Avoid relying on implicit type inference from subqueries or function return types.
6. Use meaningful table aliases (e.g., 'c' for companies, 'p' for prices)
7. For date-specific queries, use "Date"::date format when comparing with date literals
9. NEVER put quotes around expressions or comparisons. Only quote column names like "Ticker" or "Close", NOT expressions like a."Ticker" <> b."Ticker". Write them as a."Ticker" <> b."Ticker" (correct), not a."Ticker <> b."Ticker" (WRONG).
10. For comparative or ranking questions (e.g., compare multiple companies, find highest/lowest/most volatile):
- If the question asks for **"which company..."** or **"the top performer..."**, return **at least 5–10 rows** (e.g., use `LIMIT 5` or `LIMIT 10`) to allow the system to rank or analyze.
- Do NOT use `LIMIT 1`, even if the question implies "most", "highest", "lowest", etc.
- The system will handle ranking, selecting the top one, or comparing values later.
11. NEVER use window functions (e.g., LAG, LEAD, ROW_NUMBER) inside WHERE clauses. Instead, use a CTE or subquery to compute them first, then filter or sort from the outer query.
12. When using ROUND() on expressions involving double precision (like percentages, calculations, window functions), always cast the entire expression to numeric before rounding.
13. When computing percentage change over a period (e.g., 2024), select the first and last available trading prices per ticker using ORDER BY "Date" combined with DISTINCT ON ("Ticker") or subqueries. Do NOT use aggregate MIN/MAX of prices or rely solely on FIRST_VALUE/LAST_VALUE with GROUP BY.
14. When calculating percentage change over a time period (e.g., a year): Always select the first and last available trading prices for each ticker using subqueries that select MIN("Date") and MAX("Date") within the desired range. - Do NOT use "Date" = 'YYYY-MM-DD' directly, because that date might not exist due to weekends or holidays.
17: Do NOT use ^ for exponentiation in PostgreSQL. Always use the POWER(x, y) function instead.
18: When using ROUND on expressions involving POWER, division, or multiplication, cast the entire expression to numeric before passing to ROUND.
19: When using window functions (e.g., LAG, LEAD) in CTEs, always include the "Date" column in the SELECT clause if later JOINs or filters depend on it. Never assume it's present unless explicitly selected.
20. When performing arithmetic operations involving subqueries, always reference a specific column from the subquery result — never use the subquery alias directly. For example, use sub1."Close" instead of just sub1.
21. Before any division (/), explicitly cast both operands to numeric or float types to avoid implicit type mismatch errors (e.g., record / float). Do not rely on PostgreSQL to infer types automatically.
22. Avoid using reserved SQL keywords (e.g., end, start, user, order, limit, group, rank, value, etc.) as table or subquery aliases. These may cause syntax errors or unintended behavior. Use safe aliases like s, e, p, c, d, or custom names like startp, endp.
23. very important When using ROUND() on expressions involving double precision, always cast the entire expression to numeric before applying ROUND(). Avoid casting only part of the expression (e.g., 100::numeric).
""")

  # 8. Include LIMIT 5 clauses for queries that could return many rows
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
