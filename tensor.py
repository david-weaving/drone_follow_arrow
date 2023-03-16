import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



train = keras.preprocessing.image.ImageDataGenerator(rescale=1/191)
validation = keras.preprocessing.image.ImageDataGenerator(rescale=1/191)

#batch size is subject to change
train_dataset = train.flow_from_directory('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain', target_size=(200, 200), batch_size=5, class_mode='binary')
validation_dataset = train.flow_from_directory('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturesvali', target_size=(200, 200), batch_size=5, class_mode='binary')

print(train_dataset.class_indices)

model = tf.keras.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(200,200,3)),
                              tf.keras.layers.MaxPool2D(2,2),
                              tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                              tf.keras.layers.MaxPool2D(2,2),
                              tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                              tf.keras.layers.MaxPool2D(2,2),
                              # try this:
                              #tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
                              #tf.keras.layers.MaxPool2D(2,2),
                              tf.keras.layers.Flatten(),
                              tf.keras.layers.Dense(512, activation='relu'),
                              tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer= keras.optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=10, epochs=100, validation_data=validation_dataset)

if os.path.isfile('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_v1.h5') is False:
    model.save('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_v1.h5')

dir_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest'

for i in os.listdir(dir_path):
    img = keras.preprocessing.image.load_img(dir_path + '\\' + i, target_size=(200,200,3))
    plt.imshow(img)
    plt.show()
    X = keras.preprocessing.image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])

    val = model.predict(images)
    if val == 0:
        print('arrow')
        print(val)
    else:
        print('not arrow')
        print(val)
