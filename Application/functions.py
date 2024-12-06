import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# Function to compute Moving Averages
def moving_averages(data, ticker, window=5):
    ma = data[f'{ticker}_Adj Close'].rolling(window=window).mean()
    return ma

# Function to compute RSI (Relative Strength Index)
def compute_rsi(data, ticker, window=14):
    delta = data[f'{ticker}_Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute Bollinger Bands Upper
def bollinger_bands_upper(data, ticker, window=20):
    rolling_mean = data[f'{ticker}_Adj Close'].rolling(window=window).mean()
    rolling_std = data[f'{ticker}_Adj Close'].rolling(window=window).std()
    bb_upper = rolling_mean + (rolling_std * 2)
    bb_lower = rolling_mean - (rolling_std * 2)
    return bb_lower

#Function to compute Bollinger Bands Lower
def bollinger_bands_lower(data, ticker, window=20):
    rolling_mean = data[f'{ticker}_Adj Close'].rolling(window=window).mean()
    rolling_std = data[f'{ticker}_Adj Close'].rolling(window=window).std()
    bb_lower = rolling_mean - (rolling_std * 2)
    return bb_lower

# Function to compute Stochastic Oscillator K
def stochastic_oscillator_k(data, ticker, window=14):
    low_min = data[f'{ticker}_Low'].rolling(window=window).min()
    high_max = data[f'{ticker}_High'].rolling(window=window).max()
    stoch_k = 100 * (data[f'{ticker}_Adj Close'] - low_min) / (high_max - low_min)
    return stoch_k,

#Function to compute Stochastic Oscillator D
def stochastic_oscillator_d(data, ticker, window=14):
    low_min = data[f'{ticker}_Low'].rolling(window=window).min()
    high_max = data[f'{ticker}_High'].rolling(window=window).max()
    stoch_k = 100 * (data[f'{ticker}_Adj Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_d

# Function to compute On-Balance Volume (OBV)
def on_balance_volume(data, ticker):
    obv = [0]
    for i in range(1, len(data)):
        if data[f'{ticker}_Adj Close'].iloc[i] > data[f'{ticker}_Adj Close'].iloc[i - 1]:
            obv.append(obv[-1] + data[f'{ticker}_Volume'].iloc[i])
        elif data[f'{ticker}_Adj Close'].iloc[i] < data[f'{ticker}_Adj Close'].iloc[i - 1]:
            obv.append(obv[-1] - data[f'{ticker}_Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=data.index)

# Function to compute Chaikin Money Flow (CMF)
def chaikin_money_flow(data, ticker, window=20):
    money_flow_multiplier = ((data[f'{ticker}_Adj Close'] - data[f'{ticker}_Low']) - (data[f'{ticker}_High'] - data[f'{ticker}_Adj Close'])) / (data[f'{ticker}_High'] - data[f'{ticker}_Low'])
    money_flow_volume = money_flow_multiplier * data[f'{ticker}_Volume']
    cmf = money_flow_volume.rolling(window=window).sum() / data[f'{ticker}_Volume'].rolling(window=window).sum()
    return cmf

# Fetch the stock data
def fetch_last_60_days_data(ticker):
    # Get the current date
    end_date = datetime.today()
    # Calculate the start date (60 days ago)
    start_date = end_date - timedelta(days=60)
    # Format the dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    # Fetch the data
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    return data

# Example to combine all functions to return a dataset for a given ticker
def build_feature_dataset(ticker):
    data = fetch_last_60_days_data(ticker)
    
    # Get technical indicators as variables
    ma_5 = moving_averages(data, ticker, window=5)
    ma_20 = moving_averages(data, ticker, window=20)
    ma_50 = moving_averages(data, ticker, window=50)
    rsi = compute_rsi(data, ticker)
    bb_upper, = bollinger_bands_upper(data, ticker)
    bb_lower = bollinger_bands_lower(data, ticker)
    stoch_k,= stochastic_oscillator_k(data, ticker)
    stoch_d = stochastic_oscillator_d(data, ticker)
    obv = on_balance_volume(data, ticker)
    cmf = chaikin_money_flow(data, ticker)
    
    # Create a dataset with all the indicators
    dataset = pd.DataFrame({
        f'{ticker}_MA5': ma_5,
        f'{ticker}_MA20': ma_20,
        f'{ticker}_MA50': ma_50,
        f'{ticker}_RSI': rsi,
        f'{ticker}_BB_Upper': bb_upper,
        f'{ticker}_BB_Lower': bb_lower,
        f'{ticker}_StochK': stoch_k,
        f'{ticker}_StochD': stoch_d,
        f'{ticker}_OBV': obv,
        f'{ticker}_CMF': cmf
    })
    
    # Drop rows with NaN values
    dataset.dropna(inplace=True)
    
    return dataset
