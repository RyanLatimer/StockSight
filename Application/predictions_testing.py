import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import yfinance as yf
import datetime

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

# Predicting future data
def predict_future(model, last_known_data, time_steps, scaler_x, scaler_y, num_days=30):
    # Use today's date as the starting point
    start_date = datetime.datetime.now()

    # Generate future business days (skip weekends)
    predicted_dates = pd.date_range(start=start_date, periods=num_days, freq='B')

    # Only take the last `time_steps` rows of the data
    current_input = last_known_data[-time_steps:]

    predictions = []

    for _ in range(num_days):
        # Reshape current_input to (1, time_steps, num_features) for prediction
        reshaped_input = current_input.reshape(1, time_steps, current_input.shape[1])

        # Make a prediction for the next step
        prediction = model.predict(reshaped_input)

        # Inverse transform the prediction to get the predicted price change
        prediction = scaler_y.inverse_transform(prediction.reshape(-1, 1))

        # Append prediction to the list
        predictions.append(prediction[0][0])

        # Reshape the predicted value and append it to current_input
        # Convert prediction to match the feature dimensions of the input
        prediction_input = np.zeros((1, current_input.shape[1]))
        prediction_input[0, 0] = prediction[0][0]  # Place the predicted value in the first feature column

        # Update current_input with the new predicted value
        predicted_input = np.append(current_input[1:], prediction_input, axis=0)

        # Update current_input to the new predicted input
        current_input = predicted_input

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        'Date': predicted_dates,
        'Predicted_Price_Change': predictions,
    })

    # Add the predicted price change to the last known price to get predicted prices
    last_known_price = last_known_data[-1][0]  # The last price in the input data (Adj Close)
    results['Predicted_Price'] = last_known_price + np.cumsum(results['Predicted_Price_Change'])

    return results

# Define the last known data (scaled)
last_known_data = scaled_features[-time_steps:]

# Predict future data for 30 days
num_days = 30
future_predictions = predict_future(model=model, last_known_data=last_known_data, time_steps=time_steps, scaler_x=scaler_x, scaler_y=scaler_y, num_days=num_days)

# Display the predicted future prices
print(future_predictions)
