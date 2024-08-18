import tensorflow as tf
from tensorflow.keras import layers

class TicTacToeGNN(tf.keras.Model):
    def __init__(self):
        super(TicTacToeGNN, self).__init__()
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(3, 3, 1))
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(9, activation='softmax')  # 9 output units for 3x3 grid

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
