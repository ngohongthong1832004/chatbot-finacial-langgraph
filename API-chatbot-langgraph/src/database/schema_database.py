def get_schema_and_samples(conn=None):
    # Updated metadata cho database thực tế
   try:
      metadata_text = """
## Financial Database Schema for Text-to-SQL Generation

This database contains comprehensive financial and stock market data with company information, daily stock prices, and SEC filing data for financial analysis.

---

### Table: `companies`
Company master data and key metrics.

**Columns:**
* `symbol` (TEXT, PK): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
* `name` (TEXT): Full company name
* `sector` (TEXT): Business sector (e.g., 'Technology', 'Healthcare')
* `industry` (TEXT): Specific industry classification
* `country` (TEXT): Country of headquarters
* `website` (TEXT): Official company website
* `market_cap` (BIGINT): Market capitalization in USD
* `pe_ratio` (NUMERIC): Price-to-earnings ratio
* `dividend_yield` (NUMERIC): Annual dividend yield as percentage
* `fifty_two_week_high` (NUMERIC): 52-week high stock price
* `fifty_two_week_low` (NUMERIC): 52-week low stock price
* `description` (TEXT): Company description
 
 
## Available Companies (Ticker - Name)
AAPL	Apple Inc.
AMGN	Amgen Inc.
AXP	    American Express Company
BA	    Boeing Company (The)
CAT	    Caterpillar, Inc.
CRM	    Salesforce, Inc.
CSCO	Cisco Systems, Inc.
CVX	    Chevron Corporation
DIS	    Walt Disney Company (The)
DOW	    Dow Inc.
GS	    Goldman Sachs Group, Inc. (The)
HD	    Home Depot, Inc. (The)
HON	    Honeywell International Inc.
IBM	    International Business Machines
INTC	Intel Corporation
JNJ	    Johnson & Johnson
JPM	    JP Morgan Chase & Co.
KO	    Coca-Cola Company (The)
MCD	    McDonald's Corporation
MMM	    3M Company
MRK	    Merck & Company, Inc.
MSFT	Microsoft Corporation
NKE	    Nike, Inc.
PG	    Procter & Gamble Company (The)
TRV	    The Travelers Companies, Inc.
UNH	    UnitedHealth Group Incorporated
V	    Visa Inc.
VZ	    Verizon Communications Inc.
WBA	    Walgreens Boots Alliance, Inc.
WMT	    Walmart Inc.

 
**Sample Queries:**
```sql
-- Get company info
SELECT name, sector, market_cap FROM companies WHERE symbol = 'AAPL';

-- Find tech companies
SELECT symbol, name FROM companies WHERE sector = 'Technology' ORDER BY market_cap DESC;
```

---

### Table: `stock_prices`
Daily stock trading data.

**Columns:**
* `id` (SERIAL, PK): Unique identifier
* `date` (DATE): Trading date
* `open_price` (NUMERIC): Opening price
* `high_price` (NUMERIC): Daily high price
* `low_price` (NUMERIC): Daily low price
* `close_price` (NUMERIC): Closing price
* `volume` (BIGINT): Trading volume
* `dividends` (NUMERIC): Dividend amount (0 if none)
* `stock_splits` (NUMERIC): Stock split ratio (1 if none)
* `symbol` (TEXT, FK): References companies.symbol

**Sample Queries:**
```sql
-- Get recent price
SELECT date, close_price FROM stock_prices 
WHERE symbol = 'AAPL' AND date = '2024-01-15';

-- Calculate returns
SELECT symbol, 
       ((close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) / 
        LAG(close_price) OVER (PARTITION BY symbol ORDER BY date)) * 100 as daily_return
FROM stock_prices WHERE symbol = 'AAPL';
```

---

### Table: `company_cik_mapping`
Links stock symbols to SEC Central Index Keys.

**Columns:**
* `cik` (INTEGER, PK): SEC Central Index Key
* `symbol` (TEXT, FK): References companies.symbol

---

### Table: `submissions`
SEC filing metadata.

**Columns:**
* `adsh` (TEXT, PK): Accession number (filing ID)
* `cik` (INTEGER, FK): References company_cik_mapping.cik
* `form` (TEXT): Filing type ('10-K', '10-Q', '8-K', etc.)
* `period` (DATE): Reporting period end date
* `filed` (DATE): Filing date
* `accepted` (TIMESTAMP): SEC acceptance timestamp
* `fy` (INTEGER): Fiscal year
* `fp` (TEXT): Fiscal period ('FY', 'Q1', 'Q2', 'Q3', 'Q4')

**Sample Queries:**
```sql
-- Recent 10-K filings
SELECT s.adsh, c.symbol, s.form, s.filed 
FROM submissions s
JOIN company_cik_mapping ccm ON s.cik = ccm.cik
JOIN companies c ON ccm.symbol = c.symbol
WHERE s.form = '10-K' AND s.filed >= '2024-01-01';
```

---

### Table: `financial_values`
Financial metrics from SEC filings.

**Columns:**
* `id` (SERIAL, PK): Unique identifier
* `adsh` (TEXT, FK): References submissions.adsh
* `tag` (TEXT): Financial concept tag (e.g., 'Assets', 'Revenues')
* `version` (TEXT): XBRL taxonomy version
* `ddate` (DATE): Data date
* `qtrs` (INTEGER): Number of quarters (0=point-in-time, 1=quarterly, 4=annual)
* `uom` (TEXT): Unit of measure ('USD', 'shares', etc.)
* `value` (NUMERIC): Financial value
* `segments` (TEXT): Business segment info
* `coreg` (TEXT): Co-registrant info
* `footnote` (TEXT): Additional notes

**Common Financial Tags:**
- 'Assets': Total assets
- 'Revenues': Total revenues/sales
- 'NetIncomeLoss': Net income
- 'AssetsCurrent': Current assets
- 'Liabilities': Total liabilities
- 'StockholdersEquity': Shareholders' equity
- 'CashAndCashEquivalentsAtCarryingValue': Cash and equivalents

**Sample Queries:**
```sql
-- Get Apple's annual revenue
SELECT fv.ddate, fv.value as revenue
FROM financial_values fv
JOIN submissions s ON fv.adsh = s.adsh
JOIN company_cik_mapping ccm ON s.cik = ccm.cik
JOIN companies c ON ccm.symbol = c.symbol
WHERE c.symbol = 'AAPL' 
  AND fv.tag = 'Revenues'
  AND fv.qtrs = 4  -- Annual data
  AND fv.uom = 'USD'
ORDER BY fv.ddate DESC;
```

---

### Table: `tags`
Metadata for financial concept tags.

**Columns:**
* `tag` (TEXT, PK): Tag identifier
* `version` (TEXT, PK): XBRL version
* `datatype` (TEXT): Data type
* `abstract` (BOOLEAN): Is abstract concept
* `custom` (BOOLEAN): Is custom tag
* `doc` (TEXT): Tag documentation

---

### Table: `presentations`
Financial statement presentation mapping.

**Columns:**
* `adsh` (TEXT, PK): References submissions.adsh
* `report` (INTEGER, PK): Report number
* `line` (NUMERIC, PK): Line number
* `tag` (TEXT, PK): References tags.tag
* `version` (TEXT, PK): References tags.version
* `stmt` (TEXT): Statement type ('IS'=Income, 'BS'=Balance Sheet, 'CF'=Cash Flow)
* `plabel` (TEXT): Presentation label
* `negating` (BOOLEAN): Value should be negated

---

### Key Relationships and Joins

```sql
-- Complete company to stock price join
SELECT c.name, sp.date, sp.close_price
FROM companies c
JOIN stock_prices sp ON c.symbol = sp.symbol;

-- Company to financial data join
SELECT c.name, fv.tag, fv.value, fv.ddate
FROM companies c
JOIN company_cik_mapping ccm ON c.symbol = ccm.symbol
JOIN submissions s ON ccm.cik = s.cik
JOIN financial_values fv ON s.adsh = fv.adsh;

-- Complete financial analysis join
SELECT c.symbol, c.name, s.form, s.period, fv.tag, fv.value
FROM companies c
JOIN company_cik_mapping ccm ON c.symbol = ccm.symbol
JOIN submissions s ON ccm.cik = s.cik
JOIN financial_values fv ON s.adsh = fv.adsh
WHERE fv.tag IN ('Revenues', 'NetIncomeLoss', 'Assets');
```

---

### Important SQL Guidelines

1. **Column Names**: Use exact case-sensitive names as shown above
2. **Date Handling**: Always cast dates explicitly: `date::date = '2024-01-15'`
3. **Numeric Operations**: Cast to numeric for calculations: `value::numeric`
4. **Stock Symbols**: Always uppercase in queries: `symbol = 'AAPL'`
5. **Financial Tags**: Use exact tag names from XBRL taxonomy
6. **Quarters**: 0=point-in-time, 1=quarterly, 4=annual data
7. **Units**: Check `uom` column for proper unit filtering
8. **Joins**: Always use proper FK relationships as shown above

---

### Performance Tips

- Index on symbol, date for stock_prices queries
- Filter by form type early in submissions queries  
- Use qtrs and uom filters in financial_values queries
- Limit results for large datasets with LIMIT clause


### Notes







"""
      return {"metadata_text": metadata_text.strip()}
   except Exception as e:
      print(f"❌ Error loading text metadata: {e}")
      return {"error": str(e)}
