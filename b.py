import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense
from tensorflow.keras.activations import linear, softmax
from tensorflow_addons.activations import lisht
import tensorflow.keras.layers

import numpy as np

# First column.
model1 = Sequential()

model1.add(Conv2D(filters=20, kernel_size=4))

model1.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model1.add(Activation(linear))

model1.add(Conv2D(filters=40, kernel_size=5))

model1.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model1.add(Activation(linear))

model1.add(Dense(units=150, activation=lisht))

model1.add(Dense(units=10, activation=softmax))

# Second column.
model2 = Sequential()

model2.add(Conv2D(filters=20, kernel_size=4))

model2.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model2.add(Activation(linear))

model2.add(Conv2D(filters=40, kernel_size=5))

model2.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model2.add(Activation(linear))

model2.add(Dense(units=150, activation=lisht))

model2.add(Dense(units=10, activation=softmax))
