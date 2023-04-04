
# Used for training on just the hyp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os

# Training directory
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'

img_width = 960
img_height = 720
x_train = [] # images
y_train = [] # corresponding distance values

        


# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)
        distance = float(filename.split('.')[0]) # this will pull distance from the file name
        y_train.append(distance)

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
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'mse'])

model.summary()

# Shuffle training sets
train_indices = np.arange(len(x_train))
np.random.shuffle(train_indices)

# Array of images and distance values are aligned while remaining shuffled
x_train_shuffled = x_train[train_indices]
y_train_shuffled = y_train[train_indices]


# Train the model
model.fit(x_train_shuffled, y_train_shuffled, batch_size=32, epochs=30, validation_split=0.2)

model.save('D:\\models\\hyp(inverse)_model_v5.h5')
