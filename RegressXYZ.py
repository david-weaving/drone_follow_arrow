
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

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = np.array([105.2,105.2,105.2,105.2,105.2,105.2,105.2,105.2,105.2,105.2,106,106,106,106,106,106,106,106,106,106,
                    108,108,108,108,108,108,108,108,108,108,124.6,124.6,124.6,124.6,124.6,124.6,124.6,124.6,124.6,124.6,
                    160,160,160,160,160,160,177,177,177,177,177,177,177,177,177,177,179,179,179,179,179,179,179,179,179,179,180,180,180,180,180,
                    180,180,180,180,180,184,184,184,184,184,186,186,186,186,186,186,186,186,186,186,190,190,190,
                    190,190,241,241,241,241,241,294,294,294,294,294,296,296,296,296,296,296,296,296,296,296,303,303,303,
                    303,303,303,303,303,303,303,315,315,315,315,315,315,315,315,315,315,315,315,315,315,315,342,342,342,342,342,
                    342,342,342,342,342,348,348,348,348,348,348,348,348,348,348,362,362,362,362,362,362,362,362,362,362,451,451,
                    451,451,451,451,451,451,451,451,457,457,457,457,457,457,457,457,457,457,531,531,531,531,531,531,531,531,531,531,535,535,
                    535,535,535,535,535,535,535,535,535,543,543,543,543,543,543,543,543,543,543,94,94,94,94,94,94,94,94,94,94]) # IMPORTANT: make sure there are so many distance values as there are images. (x,y,z) x->distance arrow is (hyp), y->x pixel, z->ypixel

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

model.save('D:\\models\\hyp_model_v1.h5')
