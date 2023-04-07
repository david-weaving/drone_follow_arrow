import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Create train and validation data generators with random validation split
train_data = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/191,
    validation_split=0.2
)

# Load training and validation data from directory
train_dataset = train_data.flow_from_directory(
    'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary',
    subset='training'  # Use only the training subset of data for training
)
validation_dataset = train_data.flow_from_directory(
    'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain',
    target_size=(200, 200),
    batch_size=5,
    class_mode='binary',
    subset='validation'  # Use only the validation subset of data for validation
)

# Load pre-trained model
new_model = tf.keras.models.load_model('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_v1_8.h5')

# Print model summary
new_model.summary()

# Compile model
new_model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# Train model on data
model_fit = new_model.fit(
    train_dataset,
    steps_per_epoch=10,
    epochs=100,
    validation_data=validation_dataset
)

# Save the trained model
if os.path.isfile('D:\\models\\model_v1_9.h5') is False:
    new_model.save('D:\\models\\model_v1_9.h5')

# Use the trained model to predict on test images
dir_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest'

for i in os.listdir(dir_path):
    img = keras.preprocessing.image.load_img(dir_path + '\\' + i, target_size=(200, 200, 3))
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
