
#exported model works

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from djitellopy import Tello
import time
import random

model = tf.keras.models.load_model('C:\\Users\\david\\modell\\model_v1_8.h5', compile=False) #compile = False to load model

tello = Tello()
tello.connect()
battery = tello.get_battery()
print("Battery life: " , battery)

tello.streamon()

img = tello.get_frame_read().frame
new_img = cv2.imwrite(f'Resources/Images/lookanarrow.jpg',img)


img_new = tf.keras.preprocessing.image.load_img('C:\\Users\\david\\Resources\\Images\\lookanarrow.jpg', target_size=(200,200,3))
plt.imshow(img_new)
plt.show()
x = tf.keras.preprocessing.image.img_to_array(img_new)
x = np.expand_dims(x, axis=0)
img_new= np.vstack([x])
#model.predict(image here) will predict what the image is
val = model.predict(img_new)
if val == 0:
    print("I saw an arrow")
else:
    print("I did not see an arrow")

tello.streamoff()
tello.end()