import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os

# Load the model for regression
prev_model = keras.models.load_model('D:\\models\\regress_model_allthree.h5') # this previous model is good at prediction close distances, bad at far
model = keras.models.load_model('D:\\models\\regress_model_allthree_v2.h5') # this current model is good at predicting further distances, bad at close

img_width = 960
img_height = 720

# Define a function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

# Specify the path to the image
image_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\184.cm_1680984979.6978347.jpg'

# Load and preprocess the image
x = load_and_preprocess_image(image_path)

# Predict the distance, x-pixel, and y-pixel values using the loaded model
predictions = prev_model.predict(x)
predictions_new = model.predict(x)
distance, x_pixel, y_pixel = predictions
distance_new, x_pixel_new, y_pixel_new = predictions_new

print('Previous Predicted distance:', distance[0])
print('Previous Predicted x-pixel:', x_pixel[0])
print('Previous Predicted y-pixel:', y_pixel[0])

print (' ')

print('New Predicted distance:', distance_new[0])
print('New Predicted x-pixel:', x_pixel_new[0])
print('New Predicted y-pixel:', y_pixel_new[0])