import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
#loads model for regression
model = keras.models.load_model('D:\\models\\hyp(inverse)_model_v2.h5')

img_width = 960
img_height = 720

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

image_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\arrow_TESTcm_1680380816.3474493.jpg'

x = load_and_preprocess_image(image_path)

# Evaluate the model on the test set
y_pred = model.predict(x)
#y_pred = np.round(y_pred,3)

#y_pred = np.trunc(y_pred * 1000) / 1000
print(y_pred)
if np.trunc(y_pred * 10000) / 10000 <= 0.0034:
    print("Too far, fly closer")

else:
    print(1/y_pred)
