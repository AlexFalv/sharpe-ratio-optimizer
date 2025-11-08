import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical adjusted close prices for a list of tickers
    from Yahoo Finance.
    
    NOTE: yfinance default auto_adjust=True. This means it
    fetches *adjusted* close prices and names the column 'Close'.
    
    The MultiIndex structure is (PriceType, Ticker), e.g.:
    level=0: 'Open', 'High', 'Low', 'Close'
    level=1: 'AAPL', 'MSFT', 'GOOGL'
    """
    
    print("--- RUNNING THE FINAL v8 data_fetcher.py (level=0) ---")
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date
    )
    
    key_to_select = 'Close'
    
    if isinstance(data.columns, pd.MultiIndex):
        # We must select 'Close' from level=0 of the columns (axis=1).
        try:
            adj_close = data.xs(key_to_select, level=0, axis=1).copy()
        except KeyError:
            print(f"ERROR: Could not find '{key_to_select}' in MultiIndex level=0.")
            print("Available level=0 keys are:", data.columns.get_level_values(0).unique())
            raise
    else:
        # This block handles the case if you only requested one ticker
        if key_to_select in data.columns and len(tickers) == 1:
            adj_close = data[[key_to_select]].copy()
            # Rename the column from 'Close' to the ticker symbol
            adj_close.columns = tickers 
        else:
            print("ERROR: Unexpected data structure from yfinance.")
            print("DataFrame columns are:", data.columns)
            raise KeyError(f"Could not find '{key_to_select}' in yfinance data.")
            
    # Drop any rows with missing values
    adj_close.dropna(inplace=True)
    
    print("Data fetched successfully.")
    return adj_close