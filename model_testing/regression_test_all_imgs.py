import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the model for regression
prev_model = keras.models.load_model('D:\\models\\regress_model_allthree.h5')
model = keras.models.load_model('D:\\models\\regress_model_allthree_v2.h5')

img_width = 960
img_height = 720

# Define a function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# Define the path to the folder containing the images to test
folder_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'

# Loop through all images in the folder and test them
for file_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, file_name)
    
    # Load and preprocess the image
    x = load_and_preprocess_image(image_path)

    # Predict the distance, x-pixel, and y-pixel values using the loaded model
    predictions = prev_model.predict(x)
    predictions_new = model.predict(x)
    distance, x_pixel, y_pixel = predictions
    distance_new, x_pixel_new, y_pixel_new = predictions_new

    print('Image:', file_name)
    print('Previous Predicted distance:', distance[0])
    print('Previous Predicted x-pixel:', x_pixel[0])
    print('Previous Predicted y-pixel:', y_pixel[0])
    print('New Predicted distance:', distance_new[0])
    print('New Predicted x-pixel:', x_pixel_new[0])
    print('New Predicted y-pixel:', y_pixel_new[0])