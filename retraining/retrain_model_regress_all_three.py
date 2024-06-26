import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import matplotlib.pyplot as plt

# Define the directory paths for the training, validation, and testing sets
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = [] # corresponding distance, x-pixel, and y-pixel values

# Load and preprocess the training data
for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = img.astype("float32") / 255.0
        x_train.append(img)

        name_parts = filename.split('.')[0]
        distance, x_pixel, y_pixel = name_parts.split('_')
        y_train.append([float(distance), float(x_pixel), float(y_pixel)])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Load the saved model for retraining
model = keras.models.load_model('D:\\models\\regress_model_allthree.h5')
# Shuffle the indices of the training and validation sets
train_indices = np.arange(len(x_train))
np.random.shuffle(train_indices)

# Shuffle the training data while keeping the x and y elements aligned
x_train_shuffled = x_train[train_indices]
y_train_shuffled = y_train[train_indices]

# Retrain the model
history = model.fit(x_train_shuffled, {'distance': y_train_shuffled[:, 0], 'x_pixel': y_train_shuffled[:, 1], 'y_pixel': y_train_shuffled[:, 2]}, batch_size=32, epochs=100, validation_split=0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Save the retrained model
model.save('D:\\models\\regress_model_allthree_v2.h5')

# Get the training and validation loss


# Plot the training and validation loss
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()