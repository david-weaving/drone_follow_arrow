import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#code below is for retraining model

new_model = tf.keras.models.load_model('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_v1_8.h5')

new_model.summary()

dir_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest'

for i in os.listdir(dir_path):
    img = keras.preprocessing.image.load_img(dir_path + '\\' + i, target_size=(200,200,3))
    plt.imshow(img)
    plt.show()
    X = keras.preprocessing.image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])

    val = new_model.predict(images)
    if val == 0:
        print('arrow')
        print(val)
    else:
        print('not arrow')
        print(val)