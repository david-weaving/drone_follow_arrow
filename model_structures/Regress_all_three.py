import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import cv2
import os

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = [] # corresponding distance, x-pixel, and y-pixel values

# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)

        name_parts = filename.split('.')[0]
        distance, x_pixel, y_pixel = name_parts.split('_')
        y_train.append([float(distance), float(x_pixel), float(y_pixel)])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Create the model
inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
distance_output = tf.keras.layers.Dense(1, activation='linear', name='distance')(x)
x_pixel_output = tf.keras.layers.Dense(1, activation='linear', name='x_pixel')(x)
y_pixel_output = tf.keras.layers.Dense(1, activation='linear', name='y_pixel')(x)

model = keras.models.Model(inputs=inputs, outputs=[distance_output, x_pixel_output, y_pixel_output])

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

model.summary()

# Shuffle the indices of the training and validation sets
train_indices = np.arange(len(x_train))
np.random.shuffle(train_indices)

# Shuffle the training data while keeping the x and y elements aligned
x_train_shuffled = x_train[train_indices]
y_train_shuffled = y_train[train_indices]

# Train the model
model.fit(x_train_shuffled, {'distance': y_train_shuffled[:, 0], 'x_pixel': y_train_shuffled[:, 1], 'y_pixel': y_train_shuffled[:, 2]}, batch_size=32, epochs=100, validation_split=0.2)

# Save the model
model.save('E:\\models\\regress_model_allthree.h5')