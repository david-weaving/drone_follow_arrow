
#THIS TRAINS MULTIPLE IMAGES AND TESTS MULTIPLE IMAGES

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import cv2
import os

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'
val_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturesvali\\arrows'
test_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = []
y_train = np.array([5, 6]) # IMPORTANT: make sure there are so many distance values as there are images.
y_train = y_train.reshape((-1, 1))
x_val = []
y_val = np.array([4]) # IMPORTANT: make sure there are so many distance values as there are images.
y_val = y_val.reshape((-1, 1))
x_test = []
y_test = []

# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)

x_train = np.array(x_train)

# Load and preprocess the validation data
for filename in os.listdir(val_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(val_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_val.append(img)

x_val = np.array(x_val)

# Load and preprocess the testing data
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_test.append(img)

x_test = np.array(x_test)

# Create the model
model = keras.Sequential()
model.add(Rescaling(scale=1. / 255, input_shape=(img_height, img_width, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))


# Evaluate the model on the test set
y_pred = model.predict(x_test)
print(y_pred)