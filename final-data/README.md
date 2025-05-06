# DJIA Stock Price Downloader

This script downloads historical stock prices for all companies in the Dow Jones Industrial Average (DJIA) index from January 1, 2022, up to the most recent date available.

## Features

- Downloads daily stock price data (Open, High, Low, Close, Adj Close, Volume) for all 30 DJIA companies
- Implements retry logic with exponential backoff to handle API rate limits
- Saves individual CSV files for each company
- Creates a combined CSV file with all companies' data
- Provides detailed progress and summary information

## Requirements

- Python 3.7 or higher
- Required packages: yfinance, pandas, requests

## Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with Python:

```bash
python download_djia_stock_prices.py
```

## Output

The script creates a `stock_prices` directory containing:

1. Individual CSV files for each company (e.g., `AAPL_prices.csv`)
2. A combined file with all companies' data (e.g., `djia_prices_20240501.csv`)

Each file contains the following columns:
- Date: Trading date
- Open: Opening price
- High: Highest price during the day
- Low: Lowest price during the day
- Close: Closing price
- Adj Close: Adjusted closing price
- Volume: Trading volume
- Ticker: Stock symbol

## Notes

- The script uses the Yahoo Finance API via the yfinance package
- The default date range is from January 1, 2022, to the current date
- To modify the date range, edit the `start_date` and `end_date` variables in the `main()` function 

## SEC Central Index Key (CIK) Mapping

The repository includes a mapping between SEC Central Index Key (CIK) numbers and DJIA companies. This is useful for relating SEC financial filings to specific DJIA companies.

### What are CIK Numbers?

A Central Index Key (CIK) is a unique identifier assigned by the SEC to companies and individuals who file reports with the SEC. Each company that files with the SEC has at least one CIK number. Some companies may have multiple CIK numbers due to mergers, acquisitions, or other corporate events.

### Mapping Files

The repository includes the following mapping files in the `references` directory:

1. **cik_to_company_mapping.csv** - A comprehensive CSV file with the following columns:
   - CIK: The SEC Central Index Key
   - SEC_Company_Name: The official company name as registered with the SEC
   - DJIA_Ticker: The ticker symbol for matched DJIA companies
   - DJIA_Company_Name: The full name of the matched DJIA company
   - Industry: The industry classification
   - Sector: The sector classification
   - Matched: Whether the CIK was successfully matched to a DJIA company
   - Match_Score: The confidence score of the match (1.0 = perfect match)
   - Method: How the match was determined (manual_mapping, automated_matching, not_matched)

2. **cik_to_company_mapping.json** - The same mapping in JSON format for programmatic use

3. **cik_lookup.json** - A simplified lookup table for quick reference

### Key DJIA Company CIK Numbers

Here are the primary CIK numbers for select DJIA companies:

| Company | Ticker | CIK |
|---------|--------|-----|
| Apple Inc. | AAPL | 0000320193 |
| Microsoft Corp | MSFT | 0000789019 |
| JPMorgan Chase & Co | JPM | 0001090727, 0000927089 |
| Johnson & Johnson | JNJ | 0000019617 |
| Coca-Cola Co | KO | 0000021344 |
| Intel Corp | INTC | 0000066382 |
| Visa Inc | V | 0000732717 |
| Walmart Inc | WMT | 0000104169 |
| Boeing Co | BA | 0000070318 |
| Chevron Corp | CVX | 0000200406 |

### Usage

To map a CIK number to a company in Python:

```python
import json

# Load the lookup table
with open('references/cik_lookup.json', 'r') as f:
    cik_lookup = json.load(f)

# Look up a CIK
cik = '0000320193'  # Apple Inc.
if cik in cik_lookup:
    company_info = cik_lookup[cik]
    print(f"CIK {cik} maps to {company_info['name']} ({company_info['ticker']})")
else:
    print(f"CIK {cik} not found in mapping")
```

### Generating the Mapping

The mapping files were generated using the `map_cik_to_companies_fixed.py` script, which:

1. Scans all text files in the DJIA financial reports
2. Extracts unique CIK numbers
3. Uses a combination of SEC API lookups and manual mapping
4. Generates the reference files in CSV and JSON formats