
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#code below is for retraining model

new_model = tf.keras.models.load_model('D:\\models\\hyp(inverse)_model_v1.h5')

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = np.array([1/106,1/106,1/106,1/106,1/106,1/106,1/108,1/108,1/108,1/108,1/108,1/160,1/160,1/160,1/177,1/177,1/177,
                    1/180,1/180,1/180,1/180,1/180,1/180,1/186,1/186,1/186,1/186,1/186,1/186,1/190,1/190,1/190,1/241,1/241,
                    1/241,1/294,1/294,1/294,1/296,1/296,1/296,1/296,1/296,1/296,1/303,1/303,1/303,1/303,1/303,1/303,1/305,1/305,
                    1/305,1/315,1/315,1/315,1/335,1/335,1/335,1/342,1/342,1/342,1/342,1/342,1/342,1/348,1/348,1/348,1/348,
                    1/348,1/348,1/362,1/362,1/362,1/362,1/362,1/362,1/451,1/451,1/451,1/451,1/451,1/451,1/457,1/457,1/457,1/457,
                    1/457,1/457,1/531,1/531,1/531,1/531,1/531,1/531,1/535,1/535,1/535,1/535,1/535,1/535,1/543,1/543,1/543,1/543,1/543,
                    1/543,1/94,1/94,1/94]) # IMPORTANT: make sure there are so many distance values as there are images. (x,y,z) x->distance arrow is (hyp), y->x pixel, z->ypixel

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

new_model.save('D:\\models\\hyp(inverse)_model_v2.h5')