
# This will predict one image at a time
# Regression or x,y,z
# this might take a really, really long time to train
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import cv2
import os

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\david\\Resources\\Images\\train'
val_dir = 'C:\\Users\\david\\Resources\\Images\\validation'
#test_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = np.array([107,107,107,107,107,107,107,107,107,107,107,127,127,127,127]) # IMPORTANT: make sure there are so many distance values as there are images. (x,y,z) x->distance arrow is (hyp), y->x pixel, z->ypixel
x_val = []
y_val = np.array([107,107,107,107,107,107,127,127]) # IMPORTANT: make sure there are so many distance values as there are images.
x_test = []
y_test = []

# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)

x_train = np.array(x_train)

# Load and preprocess the validation data
for filename in os.listdir(val_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(val_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_val.append(img)

x_val = np.array(x_val)


# Create the model
inputs = tf.keras.layers.Input(shape=(img_height, img_width, 1))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

model.summary()

# Shuffle the indices of the training and validation sets
#train_indices = np.arange(len(x_train))
#np.random.shuffle(train_indices)
#val_indices = np.arange(len(x_val))
#np.random.shuffle(val_indices)

# Shuffle the images and corresponding labels together, but maintain their pairing
#x_train = x_train[train_indices]
#y_train = y_train[train_indices]
#x_val = x_val[val_indices]
#y_val = y_val[val_indices]

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2)


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height)) #width->960 height->720
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

model.save('C:\\Users\\david\\Resources\\Images\\models\\model.h5')
