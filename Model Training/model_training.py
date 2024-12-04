# Import necessary libraries for AI Long Short Term Memory Model
import yfinance as yf
# Torch libraries need to be installed
#import torch 
#import torch.nm as nm 
#import torch.optim 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Fetch the data from Yahoo Finance

#Define the stocks to train the model on
""" Add more stocks later.
It would be nice to train the model on 1-2k stocks across 
various sectors to build a more complete model
"""

tickers = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA', 'META', 'NVDA', 'V', 'JNJ', 'WMT', 'DIS', 'INTC', 'PYPL', 'AMD', 'NFLX', 'CSCO', 'XOM', 'PFE']

#Download all the historical data for teh last 20 years for each of the stocks
data = yf.download(tickers, start='2004-01-01', end='2024-01-01', )
print(data.head()) #Check the data

#Build moving averages

for ticker in tickers:
    data[f'{ticker}_MA5'] = data['Adj Close'][ticker].rolling(window=5).mean()
    data[f'{ticker}_MA20'] = data['Adj Close'][ticker].rolling(window=20).mean()
    data[f'{ticker}_MA50'] = data['Adj Close'][ticker].rolling(window=50).mean()


#Remove Uneccesary Rows
data.dropna
print(data.head(51))# Checl the data with 50 day MA


#Features
#Flatten the Datafram so that features are separate form labels
#A final dataset for each ticker is required

features =[]
labels = []
print("Defined features and labels array")
#Set the length for LSTM ie: How much old data to use to predict new data
sequence_length =60 #In this case 60 days of data for on day predictino

#For loop to create the feature set
for i in range(sequence_length, len(data)):
    feature_set = []
    for ticker in tickers:
        print("before extension")
        feature_set.extend([
            data['Adj Close'][ticker].iloc[i - 1],
            data[f'{ticker}_MA5'].iloc[i - 1],
            data[f'{ticker}_MA20'].iloc[i - 1],
            data[f'{ticker}_MA50'].iloc[i - 1],
            data['Volume'][ticker].iloc[i - 1]
        ])
    print("after extension")
    #Add data points to features array from feature_set
    features.append(feature_set)
    #Calculate percentage change
    close_today = data[(tickers[-1], 'Adj Close')].iloc[i-1] #Day one closing price
    close_tommorow = data[(tickers[-1], 'Adj Close')].iloc[i] #Day two closing price
    percentage_change = ((close_tommorow - close_today)/close_today)*100 #Calculate percentage change
    labels.append(ticker, percentage_change)

print("Redefining of arrays successful")

#Convert the features and labels arrays to Numpy arrays
features = np.array(features)
labels = np.array(labels)

print("Creation of NUumpy arrays succssful")