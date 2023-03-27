
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#code below is for retraining model

new_model = tf.keras.models.load_model('D:\\models\\hyp_model_v1.h5')

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = np.array([131,131,131,131,131,131,131,147,147,147,147,147,147,147,70.5,70.5,70.5,70.5,70.5,70.5,70.5,70.5]) # IMPORTANT: make sure there are so many distance values as there are images. (x,y,z) x->distance arrow is (hyp), y->x pixel, z->ypixel

# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)

x_train = np.array(x_train)

new_model.summary()

new_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'mse']) # mse added, if it doesn't work, remove it.

new_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2)

new_model.save('D:\\models\\hyp_model_v2.h5')