import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import cv2
import os
import numpy as np
#loads model for regression
model = keras.models.load_model('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_regress_v2.h5')

img_width = 960
img_height = 720

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

image_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest\\arrow_141cm_1679115611.0374367.jpg'
x = load_and_preprocess_image(image_path)

# Evaluate the model on the test set
y_pred = model.predict(x)
y_pred = np.round(y_pred)
print(y_pred)
