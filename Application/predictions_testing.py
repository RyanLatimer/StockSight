import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import functions as f
import yfinance as yf
import os
print(tf.__version__)

model = tf.keras.models.load_model('test_model.keras')
print("loading complete")
#Verify the model format
model.summary()
print("verifying complete")
#Gather user input
user_ticker = input("Enter a valid stock ticker:  ")

#Gather data from y finance
stock_data = f.fetch_last_60_days_data(user_ticker)

dataset = f.build_feature_dataset(user_ticker)

#Create the Scaler
scaler_x = MinMaxScaler(feature_range=(0,1))

#Convert to Numpy
relevant_features = ['Adj Close', 'MA5', 'MA20', 'MA50', 'return', 'Volume', 'RSI', 'BB_Upper', 'BB_Lower', 'StochK', 'StochD', 'OBV', 'CMF']
data_array = dataset[relevant_features].values

# Create sequences from the data array
time_steps = 30

def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i:i + time_steps])
    return np.array(sequences)

# Generate sequences
data_sequences = create_sequences(data_array, time_steps)

# Scale the entire data
scaled_data = scaler_x.fit_transform(data_array)

# Then create scaled sequences
scaled_sequences = create_sequences(scaled_data, time_steps)

# Create scaled sequences
scaled_sequences = create_sequences(scaled_data, time_steps)

# Make predictions
predictions = model.predict(scaled_sequences)

# If you scaled the labels during training, inverse-transform the predictions
predictions_unscaled = scaler_x.inverse_transform(predictions)

#Match predictions with dates
predicted_dates = dataset.index[time_steps - 1:]  # Adjust to match sequence length
results = pd.DataFrame({'Date': predicted_dates, 'Prediction': predictions.flatten()})
results.reset_index(drop=True, inplace=True)

# Display results
print(results)