import psycopg2
import pandas as pd

import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
 

def generate_sql_query(question):
    """Generate SQL query from natural language question using LLM"""
    # Nếu có metadata text => dùng luôn
    if get_schema_and_samples().get("metadata_text"):
        schema_description = get_schema_and_samples()["metadata_text"]
    else:
        schema_description = "No schema metadata available."
        
    prompt = ChatPromptTemplate.from_template("""
You are an expert data analyst and SQL developer. Your task is to write a correct and efficient PostgreSQL SQL query to answer the user's question using the following schema:

{schema}

🔧 Follow these strict rules:
1. Use **PostgreSQL syntax only**
2. Do **NOT** include any explanations, comments, or markdown formatting
3. If querying `djia_prices`, you must **double-quote all column names** (e.g., "Date", "Close", "Volume")
4. When filtering by date, always cast with `::date`, e.g., `"Date"::date = '2024-03-15'`
5. Use `JOIN` between `djia_companies` and `djia_prices` via `symbol = "Ticker"` when needed
6. Use proper aggregation (AVG, SUM, MAX, etc.) for time series or sector-based questions
7. When calculating returns, use:  
   `((end_price - start_price) / start_price * 100)`
8. Prefer `BETWEEN` for date ranges
9. Use `ROUND(..., 2)` when formatting percentages or decimals
10. If unsure, prefer safe and readable queries over complexity

---

User's question:  
{question}

✅ Output:
Return **only** the raw SQL query (no explanation).
""")


    # chain = prompt | llm | StrOutputParser()
    chain = prompt | RunnableLambda(call_openrouter) | StrOutputParser()

    try:
        sql_query = chain.invoke({
            "schema": schema_description,
            "question": question
        })
        return sql_query.strip()
    except Exception as e:
        print(f"❌ Error generating SQL: {e}")
        return ""
