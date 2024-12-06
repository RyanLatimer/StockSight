import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import yfinance as yf

print(tf.__version__)

# Load the model
model = tf.keras.models.load_model('test_model.keras')
model.summary()

# Gather user input
user_ticker = input("Enter a valid stock ticker: ")

# Download historical data
data = yf.download(user_ticker, start='2004-01-01', end='2024-01-01', group_by='ticker')

# Flatten the MultiIndex
data.columns = ['_'.join(col) for col in data.columns]

# Build moving averages
data[f'{user_ticker}_MA5'] = data[f'{user_ticker}_Adj Close'].rolling(window=5).mean()
data[f'{user_ticker}_MA20'] = data[f'{user_ticker}_Adj Close'].rolling(window=20).mean()
data[f'{user_ticker}_MA50'] = data[f'{user_ticker}_Adj Close'].rolling(window=50).mean()

# Add features like return and volume
data[f'{user_ticker}_return'] = data[f'{user_ticker}_Adj Close'].pct_change()
data[f'{user_ticker}_Volume'] = data[f'{user_ticker}_Volume']

# Compute RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data[f'{user_ticker}_RSI'] = compute_rsi(data[f'{user_ticker}_Adj Close'])

# Add Bollinger Bands
def bollinger_bands(data, ticker, window=20):
    rolling_mean = data[f'{ticker}_Adj Close'].rolling(window=window).mean()
    rolling_std = data[f'{ticker}_Adj Close'].rolling(window=window).std()
    data[f'{ticker}_BB_Upper'] = rolling_mean + (rolling_std * 2)
    data[f'{ticker}_BB_Lower'] = rolling_mean - (rolling_std * 2)

bollinger_bands(data, user_ticker)

# Add Stochastic Oscillator
def stochastic_oscillator(data, ticker, window=14):
    low_min = data[f'{ticker}_Low'].rolling(window=window).min()
    high_max = data[f'{ticker}_High'].rolling(window=window).max()
    stoch_k = 100 * (data[f'{ticker}_Adj Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=3).mean()
    return stoch_k, stoch_d

data[f'{user_ticker}_StochK'], data[f'{user_ticker}_StochD'] = stochastic_oscillator(data, user_ticker)

# Add On-Balance Volume (OBV)
def on_balance_volume(data, ticker):
    obv = [0]
    for i in range(1, len(data)):
        if data[f'{ticker}_Adj Close'].iloc[i] > data[f'{ticker}_Adj Close'].iloc[i - 1]:
            obv.append(obv[-1] + data[f'{ticker}_Volume'].iloc[i])
        elif data[f'{ticker}_Adj Close'].iloc[i] < data[f'{ticker}_Adj Close'].iloc[i - 1]:
            obv.append(obv[-1] - data[f'{ticker}_Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv

data[f'{user_ticker}_OBV'] = on_balance_volume(data, user_ticker)

# Add Chaikin Money Flow (CMF)
def chaikin_money_flow(data, ticker, window=20):
    adl = (2 * data[f'{ticker}_Adj Close'] - data[f'{ticker}_Low'] - data[f'{ticker}_High']) / (data[f'{ticker}_High'] - data[f'{ticker}_Low'])
    adl = adl * data[f'{ticker}_Volume']
    cmf = adl.rolling(window=window).sum() / data[f'{ticker}_Volume'].rolling(window=window).sum()
    data[f'{ticker}_CMF'] = cmf

chaikin_money_flow(data, user_ticker)

# Drop NaN values
data.dropna(inplace=True)

# Define relevant features and target variable
relevant_features = [f'{user_ticker}_Adj Close', f'{user_ticker}_MA5', f'{user_ticker}_MA20', f'{user_ticker}_MA50', 
                     f'{user_ticker}_return', f'{user_ticker}_Volume', f'{user_ticker}_RSI', f'{user_ticker}_BB_Upper', 
                     f'{user_ticker}_BB_Lower', f'{user_ticker}_StochK', f'{user_ticker}_StochD', f'{user_ticker}_OBV', 
                     f'{user_ticker}_CMF']

# Target variable is the 'Adj Close' column
target_column = f'{user_ticker}_Adj Close'

# Extract features and target data
features_data = data[relevant_features].values
target_data = data[target_column].values.reshape(-1, 1)

# Scale features
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_x.fit_transform(features_data)

# Scale target variable
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_y.fit_transform(target_data)

# Create sequences for training
time_steps = 30
def create_sequences(data, time_steps):
    sequences = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
    return np.array(sequences)

scaled_feature_sequences = create_sequences(scaled_features, time_steps)

# Train your model using `scaled_feature_sequences` and `scaled_target`
# model.fit(scaled_feature_sequences, scaled_target_sequences)

# Predicting future data
def predict_future(model, last_known_data, time_steps, scaler_x, scaler_y):
    # Initialize the predictions list
    predictions = []

    # Use the last known data for predictions
    current_input = last_known_data

    # Predict the future one step at a time
    for _ in range(30):  # Predict the next 30 days (for example)
        # Make a prediction
        prediction = model.predict(current_input.reshape(1, time_steps, -1))  # Shape must be (1, time_steps, features)
        
        # Append prediction to the list
        predictions.append(prediction[0, 0])

        # Create a new input by appending the predicted value
        # Create a placeholder with zeros for the new features (same number of features as in training)
        new_row = np.zeros((1, current_input.shape[1]))  # Shape (1, features)
        new_row[0, 0] = prediction  # Assign the predicted value to the first feature
        
        # Slide the window by removing the first row and adding the new row
        current_input = np.append(current_input[1:], new_row, axis=0)

    # Inverse transform the predictions to get actual values
    predictions_unscaled = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions_unscaled


# Get the last known data for prediction
last_known_data = scaled_features[-time_steps:]

# Predict the next 30 days of stock prices
future_predictions = predict_future(model, last_known_data, time_steps, scaler_x, scaler_y)

# Display results
predicted_dates = pd.date_range(start=data.index[-1], periods=31, freq='B')[1:]  # 30 business days
results = pd.DataFrame({
    'Date': predicted_dates,
    'Predicted_Price': future_predictions.flatten()
})
results.reset_index(drop=True, inplace=True)

print(results)
