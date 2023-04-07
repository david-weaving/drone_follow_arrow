import moving_calculations as mc
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from djitellopy import Tello

tello = Tello()
#tello.connect()
#battery = tello.get_battery()
#print("Battery life: " , battery)

#tello.streamon()

#img = tello.get_frame_read().frame
#new_img = cv2.imwrite(f'[INSERT IMAGE PATH]',img)

img_width = 960
img_height = 720

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

prev_model = keras.models.load_model('E:\\models\\regress_model_allthree.v2.1.h5')

#tello.takeoff()

#img = tello.get_frame_read().frame
#new_img = cv2.imwrite(f'[INSERT IMAGE PATH]',img)

image_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\165_173_171.cm_1680733473 (2).jpg'
x = load_and_preprocess_image(image_path)

predictions = prev_model.predict(x)
distance, x_pixel, y_pixel = predictions
#print('Predicted distance:', distance[0])
#print('Predicted x-pixel:', x_pixel[0])
#print('Predicted y-pixel:', y_pixel[0])

z, x, y = mc.find_direction(x_pixel[0], y_pixel[0], distance[0])
z = np.round(z)
x = np.round(x)
y = np.round(y)
#print(f"The arrow is {z}cm in the z direction, {y}cm in the y direction, and {x}cm in the x direction.")

if x < 0:
    x = x*-1
elif x > 0:
    tello.move_right(x)

if y < 0:
    y = y*-1
    tello.move_down(y)
elif y > 0:
    tello.move_up(y)

tello.move_forward(z)