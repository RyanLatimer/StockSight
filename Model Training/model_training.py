# Import necessary libraries for AI Long Short Term Memory Model
import yfinance as yf
import torch # type: ignore
import torch.nm as nm # type: ignore
import torch.optim # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Fetch the data from Yahoo Finance

#Define the stocks to train the model on
""" Add more stocks later.
It would be nice to train the model on 1-2k stocks across 
various sectors to build a more complete model
"""

tickers = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'META', 'NVDA', 'V', 'JNJ', 'WMT', 'DIS', 'INTC', 'PYPL', 'AMD', 'NFLX', 'CSCO', 'XOM', 'PFE']

# Download all the historical data for teh last 20 years for each of the stocks
data = yf.download(tickers, start='2004-01-01', end='2024-01-01', )
print(data.head()) #Check the data

#Build moving averages
for ticker in tickers:
    data[('Adj CLose', 'MA5')] = data[('Adj Close', ticker)].rolling(window=5).mean()
    data[('Adj CLose', 'MA20')] = data[('Adj Close', ticker)].rolling(window=20).mean()
    data[('Adj CLose', 'MA50')] = data[('Adj Close', ticker)].rolling(window=50).mean()

#Remove Uneccesary Rows
data.dropna
print(data.head(51))# Checl the data with 50 day MA


