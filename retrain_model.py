import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

#code below is for retraining model

new_model = tf.keras.models.load_model('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_v1_8.h5')

train = keras.preprocessing.image.ImageDataGenerator(rescale=1/191)
validation = keras.preprocessing.image.ImageDataGenerator(rescale=1/191)

#batch size is subject to change
train_dataset = train.flow_from_directory('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain', target_size=(200, 200), batch_size=5, class_mode='binary')
validation_dataset = train.flow_from_directory('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturesvali', target_size=(200, 200), batch_size=5, class_mode='binary')

new_model.summary()

new_model.compile(loss='binary_crossentropy', optimizer= keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

model_fit = new_model.fit(train_dataset, steps_per_epoch=10, epochs=100, validation_data=validation_dataset)

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
    
