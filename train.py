import numpy as np
import tensorflow as tf
from model import TicTacToeGNN

# Example training data
X_train = np.random.random((1000, 3, 3, 1))  # Shape: (num_samples, height, width, channels)
y_train = np.random.random((1000, 9))        # Shape: (num_samples, num_classes)

model = TicTacToeGNN()
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)

# Save the model weights
model.save_weights('model_weights.h5')
