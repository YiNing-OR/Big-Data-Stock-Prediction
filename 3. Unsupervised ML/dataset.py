import pandas as pd

class StockDataset():
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        """
        Implement a pipeline for cleaning the stock market dataset
        """
        # Convert date to datetime object
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Sort entries by date (earliest on top)
        self.df = self.df.sort_values(by='date', ignore_index=True)

    def compute_pct_change(self, column='close'):
        """
        Compute row-to-row percentage change in price for a given column

        Args:
            column (str): Column name to compute the data from
        """
        self.df[f'{column}]]_pct_change'] = self.df[column].pct_change().mul(100).round(2)

    def compute_ema(self, duration):
        """
        Compute the Exponential Moving Averages using the closing price

        Args:
            duration (int): duration to compute the EMA
        """
        self.df[f'ema{duration}'] = self.df['close'].ewm(span=duration, adjust=False).mean()



