import tensorflow as tf
import numpy as np
import pandas as pd
import functions as f
import yfinance as yf

#Load the model
model = tf.keras.models.load_model('../Model/multi_industry_full_technical.keras')

#Gather user input
user_ticker = input("Enter a valid stock ticker:  ")

#Gather data from y finance
data = yf.download(user_ticker, start='2004-01-01', end='2024-01-01', group_by='ticker' )

#Build the input dataset
MA5 = f.moving_averages(data, user_ticker, window=5)
MA20 = f.moving_averages(data, user_ticker, window=20)
MA50 = f.moving_averages(data, user_ticker, window=50)
RSI = f.compute_rsi(data, user_ticker,)
BB_lower = f.bollinger_bands_lower(data, user_ticker)
BB_upper = f.bollinger_bands_upper(data, user_ticker)
Stoch_k = f.stochastic_oscillator_k(data, user_ticker, 14)
Stock_d = f.stochastic_oscillator_d(data, user_ticker, 14)
OBV = f.on_balance_volume(data, user_ticker,)