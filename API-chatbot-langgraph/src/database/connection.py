import psycopg2
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import requests
from src.database.schema_database import get_schema_and_samples
from sqlalchemy import create_engine

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
    }


def get_db_uri():
    return f"postgresql+psycopg2://{os.getenv('DBUSER')}:{os.getenv('DBPASSWORD')}@{os.getenv('DBHOST')}:{os.getenv('DBPORT')}/{os.getenv('DBNAME')}"


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
        engine = create_engine(get_db_uri())
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"❌ SQL execution error: {e}")
        return None
    
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

    # Get schema information
    sample_data = get_schema_and_samples(conn=None)
    sample_description = sample_data.get("metadata_text", "")

    # Updated comprehensive prompt for financial database
    prompt = ChatPromptTemplate.from_template("""
You are an expert PostgreSQL developer specializing in financial data analysis and SEC filings.

# Database Schema
{schema}

# User Question
{question}

# Critical Instructions

## Query Structure
1. **Return ONLY the SQL query** - no explanations, markdown, or formatting
2. **Use exact table and column names** from the schema above
3. **Always use proper PostgreSQL syntax** with explicit type casting

## Table-Specific Rules

### Companies Table
- Primary key: `symbol` (TEXT)
- Use `symbol = 'AAPL'` format (uppercase symbols)
- Available metrics: market_cap, pe_ratio, dividend_yield, etc.

### Stock_prices Table  
- Date column: `date` (DATE type)
- Price columns: open_price, high_price, low_price, close_price (NUMERIC)
- Use `date::date = '2024-01-15'` for date comparisons
- Use `BETWEEN '2024-01-01' AND '2024-12-31'` for date ranges
- Join with companies: `JOIN companies c ON sp.symbol = c.symbol`

### Financial Data (SEC Filings)
- Complex join path: companies → company_cik_mapping → submissions → financial_values
- Common tags: 'Revenues', 'NetIncomeLoss', 'Assets', 'StockholdersEquity'
- Filter by qtrs: 0=point-in-time, 1=quarterly, 4=annual
- Filter by uom: 'USD' for monetary values
- Use `fv.value::numeric` for calculations

## Data Type Handling
- **Always cast before arithmetic**: `(close_price::numeric - open_price::numeric)`
- **Date casting**: `date::date = '2024-01-15'`
- **Numeric calculations**: Cast all operands to `::numeric` or `::float`
- **Division safety**: Cast both numerator and denominator before division

## Financial Calculations
- **Returns**: `((end_price::numeric - start_price::numeric) / start_price::numeric) * 100`
- **Volatility**: Use `STDDEV()` function with proper casting
- **Moving averages**: Use window functions with `ROWS BETWEEN`
- **Year-over-year**: Join same table with different periods
- **Equal-weighted Total Return (DJIA-style)**:
  • Use a WITH clause to extract:
     - `start_prices`: SELECT symbol, open_price WHERE date = '2024-01-02'
     - `end_prices`: SELECT symbol, close_price WHERE date = '2024-12-31'
  • Join `start_prices` and `end_prices` ON symbol
  • Compute return per stock: `(end_price - start_price) / start_price * 100`
  • Return the average return using: `AVG(...) AS djia_total_return`

## Common Query Patterns

### Stock Price Analysis
```sql
-- Price lookup
SELECT date, close_price FROM stock_prices 
WHERE symbol = 'AAPL' AND date::date = '2024-01-15';

-- Returns calculation  
SELECT symbol, date,
       ((close_price::numeric - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date))
        / LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) * 100 as daily_return
FROM stock_prices WHERE symbol = 'AAPL';
```

### Financial Statement Data
```sql
-- Annual revenue
SELECT c.symbol, fv.ddate, fv.value::numeric as revenue
FROM companies c
JOIN company_cik_mapping ccm ON c.symbol = ccm.symbol  
JOIN submissions s ON ccm.cik = s.cik
JOIN financial_values fv ON s.adsh = fv.adsh
WHERE fv.tag = 'Revenues' AND fv.qtrs = 4 AND fv.uom = 'USD';
```

## Window Functions & Aggregations
- **Never use window functions in WHERE clauses**
- **Use CTEs for complex window function logic**
- **Always include necessary columns in SELECT when using window functions**
- **Use proper PARTITION BY and ORDER BY clauses**

## Performance & Results
- **Add LIMIT clauses** for queries that may return many rows
- **Use meaningful table aliases**: c=companies, sp=stock_prices, fv=financial_values
- **For ranking questions**: Return 5-10 results, not just top 1
- **Avoid reserved keywords** as aliases (end, start, value, etc.)

## Error Prevention
- **Never use ^ for exponentiation** - use POWER(base, exponent)
- **Cast expressions before ROUND()**: `ROUND(expression::numeric, 2)`
- **Reference specific columns** from subqueries, not just aliases
- **Use safe aliases** to avoid SQL keyword conflicts
- **Always cast both operands** in division operations
- **Avoid using reserved SQL keywords** (such as `end`, `start`, `value`, `date`, `group`, `order`, `rank`) as column aliases, table aliases, or variable names.
- **Instead, use safe alternatives** like `end_date`, `start_price`, `metric_value`, `ranking`, etc.


## Question-Specific Logic
- **Comparative questions**: Return multiple rows for analysis
- **Time series**: Use proper date ordering and window functions
- **Financial ratios**: Ensure both numerator and denominator are cast to numeric
- **Period comparisons**: Use appropriate qtrs filtering (1=quarterly, 4=annual)
- **Multi-company analysis**: Include company names and symbols in results

## Common SQL Patterns for Financial QA

### 1. Price and Volume Lookup (Factual)
- Closing Price:  
  `SELECT close_price FROM stock_prices WHERE symbol = 'XYZ' AND date::date = 'YYYY-MM-DD';`

- Opening Price:  
  `SELECT open_price FROM stock_prices WHERE symbol = 'XYZ' AND date::date = 'YYYY-MM-DD';`

- High / Low Price:  
  `SELECT high_price, low_price FROM stock_prices WHERE symbol = 'XYZ' AND date::date = 'YYYY-MM-DD';`

- Trading Volume:  
  `SELECT volume FROM stock_prices WHERE symbol = 'XYZ' AND date::date = 'YYYY-MM-DD';`

### 2. Extremes (Highest / Lowest) With Date
- Highest Closing Price:  
  `SELECT date, close_price FROM stock_prices WHERE symbol = 'XYZ' AND close_price = (SELECT MAX(close_price) FROM stock_prices WHERE symbol = 'XYZ' AND date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD');`

- Lowest Closing Price:  
  `... MIN(close_price) ...`

### 3. Dividend Queries
- Dividend Amount on Date:  
  `SELECT dividends FROM stock_prices WHERE symbol = 'XYZ' AND date::date = 'YYYY-MM-DD';`

- Total Dividend Count:  
  `SELECT COUNT(*) FROM stock_prices WHERE symbol = 'XYZ' AND dividends > 0 AND date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD';`

### 4. Return-Based Queries
- Percentage Return over Period:  
  `((end_price - start_price) / start_price * 100)`

- Total Return for DJIA (Equal-Weighted):  
  Use `WITH start_prices` and `end_prices`, then `AVG(...)` across symbols

### 5. Drawdown
- Maximum Drawdown:  
```sql
WITH peaks AS (
  SELECT date, close_price,
         MAX(close_price) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) AS peak
  FROM stock_prices WHERE symbol = 'XYZ' AND date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
)
SELECT MAX((peak - close_price) / peak * 100) FROM peaks;

6. Average / Aggregates
- Average Closing Price in Month/Qtr:  
  SELECT AVG(close_price) FROM stock_prices WHERE symbol = 'XYZ' AND date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD';
- Median Price:  
  SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY close_price) FROM stock_prices WHERE symbol = 'XYZ';
- Average Volume:  
  SELECT AVG(volume) FROM stock_prices WHERE symbol = 'XYZ';
  
7. Volatility and Risk
- Daily Return:((close_price - LAG(close_price)) / LAG(close_price))
- Daily Volatility:STDDEV(daily_return)
- Annualized Volatility:STDDEV(daily_return) * SQRT(252)
- Beta:COV(stock_return, market_return) / VAR(market_return)
- Correlation:SELECT CORR(a.daily_return, b.daily_return) ...

8. Threshold Counting
- Days Above Threshold:SELECT COUNT(*) FROM stock_prices WHERE symbol = 'XYZ' AND close_price > VALUE;
- Percent of Days:(COUNT(condition)::float / COUNT(*)) * 100
- Within 1 Std Dev:ABS(close_price - mean) <= stddev

9. Sorting & Ranking
- Top Performers:Use RANK() OVER (ORDER BY total_return DESC)
- Highest Volume Day:ORDER BY volume DESC LIMIT 1

10. Chart-Specific
- Line Plot: SELECT date, close_price ... ORDER BY date
- Histogram: SELECT ((close_price - LAG(close_price)) / LAG(close_price)) AS daily_return ...
- Boxplot: group by month or quarter and use closing price
- Scatter Plot: use two metrics like market_cap vs pe_ratio
- Pie Chart: group by sector and aggregate (e.g. SUM(market_cap))
- For correlation heatmaps: compute daily returns for multiple symbols and use `SELECT CORR(a.daily_return, b.daily_return)` in a JOIN query aligned by date.
- For boxplots grouped by month, return one row per day with: 
  `SELECT DATE_TRUNC('month', date) AS month, close_price ...` 
  then group in Python using `.groupby('month')['close_price'].apply(list)`
- NEVER use: `ARRAY_AGG(close_price)` — this makes the output harder to use for plotting.
- Do NOT compute correlation directly in SQL using CORR(...).
- Instead, select raw daily returns by symbol and date. Use pandas to pivot into a matrix and calculate `.corr()` before plotting the heatmap.

## Chart Generation Data Requirements
- If the user question involves visualization, plotting, charting, drawing, or trend analysis (including keywords such as: "plot", "chart", "draw", "line chart", "boxplot", "heatmap", "visualize", "trend", "scatter", "compare over time"):
  - Always return raw and granular data suitable for direct plotting.
  - NEVER return aggregated arrays or precomputed correlations (e.g., ARRAY_AGG(), CORR()).
  - Instead, return one row per observation (e.g., per date per symbol).
  - For line charts or trend plots: `SELECT date, close_price FROM stock_prices ... ORDER BY date`
  - For boxplots grouped by month: return `DATE_TRUNC('month', date)` and `close_price` per row — then group in Python.
  - For heatmaps of correlation: return raw `daily_return` per `symbol` and `date`, then compute `.pivot().corr()` in Python.
  - Do NOT do any reshaping or matrix-like output in SQL — let Python handle reshaping and chart rendering.
  - Ensure all needed fields (e.g., date, symbol, metric) are present for plotting, not just summary statistics.

## Additional Chart & Subquery Instructions
- Always add `ORDER BY date` when plotting time series.
- Group by `DATE_TRUNC('month', date)` or `EXTRACT(MONTH FROM date)` for monthly plots.
- When joining or using subqueries, reference columns with explicit aliases (e.g., `sp.close_price`, `c.sector`).
- For correlation heatmaps, use `CORR(...)` across aligned returns of multiple tickers.
- Do NOT use ARRAY_AGG() to group values into arrays. Always return one row per observation (e.g., per date or per company).
- For boxplot or time series, return raw rows (e.g., `SELECT date, close_price`) so that charts can be drawn directly without parsing or exploding.

Generate the SQL query following these guidelines exactly.

""")

    # Create the chain
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
