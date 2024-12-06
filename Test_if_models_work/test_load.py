import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load the model
loaded_model = tf.keras.models.load_model('simple_model.keras')

# Generate dummy input data (10 features)
test_input = np.random.rand(1, 10)  # 1 sample with 10 features

# Make a prediction
prediction = loaded_model.predict(test_input)

print("Input:", test_input)
print("Prediction:", prediction)

