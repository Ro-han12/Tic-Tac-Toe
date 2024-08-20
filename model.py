import numpy as np
import tensorflow as tf
from model import TicTacToeGNN

# Example corrected training data
# Random board states (3x3 grid, single channel)
X_train = np.random.randint(0, 2, (1000, 3, 3, 1)).astype('float32')  # Convert to float32
# Random correct moves (one-hot encoded)
y_train = np.zeros((1000, 9))
for i in range(1000):
    y_train[i, np.random.randint(0, 9)] = 1  # Random valid move

# Initialize and compile the model
model = TicTacToeGNN()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=80)

# Save the model weights
model.save_weights('model_weights_epochs80.h5')
