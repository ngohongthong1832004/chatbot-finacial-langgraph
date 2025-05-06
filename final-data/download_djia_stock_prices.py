import yfinance as yf
import pandas as pd
import time
import random
import os
from datetime import datetime, date
from requests.exceptions import HTTPError

def get_djia_constituents():
    """
    Get list of companies in the DJIA index.
    
    The DJIA (Dow Jones Industrial Average) ticker symbol is ^DJI
    """
    # We can get DJIA constituents using the following method
    djia = yf.Ticker("^DJI")
    
    try:
        # Try to get constituents directly (may not always work)
        return djia.constituents
    except:
        # Fallback: hardcoded list of DJIA companies as of 2023
        # This list can change over time, so it's better to fetch it dynamically if possible
        djia_tickers = [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
            "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
        ]
     
        return djia_tickers

def download_stock_prices_with_retry(ticker, start_date, end_date, max_retries=5, initial_delay=2):
    """
    Download historical stock prices for a given ticker with retry logic.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_retries (int): Maximum number of retry attempts
        initial_delay (int): Initial delay between retries in seconds
        
    Returns:
        DataFrame: Historical stock price data
    """
    for attempt in range(max_retries):
        try:
            # Add a delay between attempts that increases with each retry
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f"Retry {attempt}/{max_retries-1} for {ticker}. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                
            # Instead of using yf.download directly, create a Ticker object first
            print(f"Downloading {ticker} data (attempt {attempt+1}/{max_retries})...")
            ticker_obj = yf.Ticker(ticker)
            
            # Get the historical data using the history method
            # Use period='max' and filter later for better reliability
            data = ticker_obj.history(period="2y")  # Get 2 years of data to ensure we cover our range
            
            # Check if we got data
            if data.empty:
                print(f"No data returned for {ticker} on attempt {attempt+1}")
                continue
                
            # Filter to the requested date range
            filtered_data = data.loc[start_date:end_date] if not data.empty else pd.DataFrame()
            
            if filtered_data.empty:
                print(f"No data available for {ticker} in the specified date range")
                # Try a different approach with direct download
                try:
                    # Try using a longer time range and then filter
                    from datetime import datetime, timedelta
                    
                    # Convert start_date to datetime and go back 30 days
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30)
                    extended_start = start_dt.strftime("%Y-%m-%d")
                    
                    direct_data = yf.download(
                        ticker,
                        start=extended_start,
                        end=end_date,
                        interval="1d",
                        progress=False
                    )
                    
                    if not direct_data.empty:
                        # Filter to original date range
                        filtered_direct = direct_data.loc[start_date:end_date] if len(direct_data) > 0 else pd.DataFrame()
                        
                        if not filtered_direct.empty:
                            filtered_direct['Ticker'] = ticker
                            filtered_direct = filtered_direct.reset_index()
                            print(f"Successfully downloaded {ticker} data with extended date method")
                            return filtered_direct
                    
                    # If that failed, try a different approach with 1mo interval
                    print(f"Trying with monthly data for {ticker}...")
                    monthly_data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        interval="1mo",  # Try monthly data
                        progress=False
                    )
                    
                    if not monthly_data.empty:
                        monthly_data['Ticker'] = ticker
                        monthly_data = monthly_data.reset_index()
                        print(f"Successfully downloaded {ticker} monthly data")
                        return monthly_data
                        
                except Exception as direct_e:
                    print(f"Direct download failed for {ticker}: {direct_e}")
                    # Continue to next attempt
            else:
                # Add ticker column to the filtered data
                filtered_data['Ticker'] = ticker
                
                # Reset index to make Date a column
                filtered_data = filtered_data.reset_index()
                
                print(f"Successfully downloaded {ticker} data with {len(filtered_data)} records")
                return filtered_data
                
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Error downloading {ticker} (attempt {attempt+1}): {e}")
            
            # If it's a JSON parsing error, wait longer before retrying
            if "json" in error_msg or "expecting value" in error_msg:
                time.sleep(5)  # Wait longer for API issues
    
    # If we get here, all attempts failed
    print(f"Failed to download data for {ticker} after {max_retries} attempts")
    
    # Create an empty dataframe with the expected columns as a fallback
    empty_df = pd.DataFrame({
        'Date': [],
        'Open': [],
        'High': [], 
        'Low': [],
        'Close': [],
        'Volume': [],
        'Dividends': [],
        'Stock Splits': [],
        'Ticker': []
    })
    
    return empty_df

def main():
    # Define the date range
    start_date = "2022-01-01"
    end_date = date.today().strftime('%Y-%m-%d')  # Today's date
    
    print(f"Downloading stock prices from {start_date} to {end_date}")
    
    # Create output directory if it doesn't exist
    output_dir = "stock_prices"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fetching DJIA constituents...")
    tickers = get_djia_constituents()
    
    if not tickers:
        print("Failed to retrieve DJIA constituents.")
        return
    
    print(f"Found {len(tickers)} companies in the DJIA index.")
    
    # Get all stock prices
    all_prices = pd.DataFrame()
    
    for i, ticker in enumerate(tickers, 1):
        print(f"Downloading price data for {ticker} ({i}/{len(tickers)})...")
        
        # Download with retry logic
        prices = download_stock_prices_with_retry(ticker, start_date, end_date)
        
        if not prices.empty:
            # Append to the main DataFrame
            all_prices = pd.concat([all_prices, prices])
            
            # Also save individual stock data
            ticker_file = os.path.join(output_dir, f"{ticker}_prices.csv")
            prices.to_csv(ticker_file, index=False)
            print(f"  - Saved {ticker} data to {ticker_file}")
        
        # Add a delay between requests to avoid rate limiting
        time.sleep(1.5)
    
    # If we have data, save the combined file
    if not all_prices.empty:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d')
        combined_file = os.path.join(output_dir, f"djia_prices_{timestamp}.csv")
        
        # Save combined data
        all_prices.to_csv(combined_file, index=False)
        print(f"\nAll stock prices saved to {combined_file}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  - Total companies processed: {len(tickers)}")
        print(f"  - Companies with data: {all_prices['Ticker'].nunique()}")
        print(f"  - Total records: {len(all_prices):,}")
        print(f"  - Date range: {all_prices['Date'].min()} to {all_prices['Date'].max()}")
    else:
        print("No data was downloaded successfully.")

if __name__ == "__main__":
    main() 