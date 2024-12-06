import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Create a simple model with explicit input shape
simple_model = Sequential([
    Input(shape=(10,)),  # Input has 10 features
    Dense(10, activation='relu'),
    Dense(1)  # Single output
])
simple_model.compile(optimizer=Adam(), loss='mse')

# Save the model
simple_model.save('simple_model.keras')


