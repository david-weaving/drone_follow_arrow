
# This will predict one image at a time
# Regression or x,y,z
# this might take a really, really long time to train
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import cv2
import os
import matplotlib as plt

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'
arrow_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\unfiltered_arrow_pics' # holds

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = np.array([1/112,1/112,1/112,1/112,1/112,1/112,1/148,1/148,1/148,1/167,1/167,1/167,1/167,1/167,1/221,1/221,1/221,1/221,1/221,1/221,1/226,1/226,1/226,1/226,1/226,1/226,
                    1/243,1/243,1/243,1/243,1/243,1/243,1/248,1/248,1/248,1/248,1/248,1/248,1/285,1/285,1/285,1/285,1/285,1/285,1/348,1/348,1/348,1/348,1/348,1/348,1/351,1/351,1/351,1/351,1/351,1/351,
                    1/362,1/362,1/362,1/362,1/362,1/385,1/385,1/385,1/385,1/385,1/385,1/410,1/410,1/410,1/410,1/410,1/410,1/85,1/85,1/85]) # IMPORTANT: make sure there are so many distance values as there are images. (x,y,z) x->distance arrow is (hyp), y->x pixel, z->ypixel

        


# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)

x_train = np.array(x_train)

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
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'mse']) # mse added, if it doesn't work, remove it.

model.summary()

# Shuffle the indices of the training and validation sets
train_indices = np.arange(len(x_train))
np.random.shuffle(train_indices)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2)

model.save('D:\\models\\hyp(inverse)_model_v2.h5')
