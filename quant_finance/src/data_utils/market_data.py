import pandas as pd
import yfinance as yf
from typing import List, Optional, Union
from datetime import datetime, timedelta

class MarketDataFetcher:
    """A class to fetch and process financial market data."""
    
    def __init__(self):
        """Initialize the MarketDataFetcher."""
        self.data = {}
    
    def fetch_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch historical stock data for given symbols.
        
        Args:
            symbols: Single stock symbol or list of symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            DataFrame with historical price data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
        dfs = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            df['Symbol'] = symbol
            dfs.append(df)
            
        return pd.concat(dfs) if len(dfs) > 1 else dfs[0]
    
    def calculate_returns(
        self,
        data: pd.DataFrame,
        method: str = 'simple'
    ) -> pd.DataFrame:
        """Calculate returns from price data.
        
        Args:
            data: DataFrame with 'Close' price column
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame with calculated returns
        """
        if method == 'simple':
            returns = data['Close'].pct_change()
        elif method == 'log':
            returns = np.log(data['Close'] / data['Close'].shift(1))
        else:
            raise ValueError("method must be 'simple' or 'log'")
            
        return returns
    
    def calculate_volatility(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """Calculate rolling volatility of returns.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Series with rolling volatility
        """
        return returns.rolling(window=window).std() * np.sqrt(window)