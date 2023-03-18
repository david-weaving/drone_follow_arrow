
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
train_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows'
val_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturesvali\\arrows'
test_dir = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestest'

# Define the image dimensions and load and preprocess the data
img_width = 960
img_height = 720
x_train = [] # images
y_train = np.array([(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),(70.5, 225, 84),
                    (107,434,225),(107,434,225),(107,434,225), (131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),
                    (131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462),(131,294,462), (141,574,178), (147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),
                    (147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(147,410,246),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),
                    (154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(154,298,247),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),(178,643,58),
                    (178,643,58),(178,643,58),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),(189.5,537,218),
                    (189.5,537,218),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),(392,393,231),
                    (392,393,231)]) # IMPORTANT: make sure there are so many distance values as there are images. (x,y,z) x->distance arrow is (hyp), y->x pixel, z->ypixel
x_val = []
y_val = np.array([(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),
                  (94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96),(94,557,96), (105.2,624,280),(105.2,624,280),(105.2,624,280),(105.2,624,280),(124.6,407,313),(124.6,407,313),(124.6,407,313),(124.6,407,313),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),
                  (177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(177,646,350),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),(179,396,1),
                  (179,396,1),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),(335.5,534,269),
                  (335.5,534,269),(335.5,534,269)]) # IMPORTANT: make sure there are so many distance values as there are images.
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
model = keras.Sequential()
model.add(Rescaling(scale=1. / 255, input_shape=(img_height, img_width, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3))

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(), loss='mean_squared_error', metrics=['mae'])

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
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

model.save('C:\\Users\\Administrator\\PycharmProjects\\HellWord\\models\\model_regress_v2.h5')

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

#image_path = 'C:\\Users\\Administrator\\PycharmProjects\\HellWord\\picturestrain\\arrows\\32cm_1678502452.7660737.jpg'
#x = load_and_preprocess_image(image_path)

# Evaluate the model on the test set
#pred = model.predict(x)
#print(pred)

#z_pred = model.predict(x)[0,2] #predicts y-pixel value
#y_pred = model.predict(x)[0,1] #predicts x-pixel value
#x_pred = model.predict(x)[0,0] #predicts hyptonuse value

#print(z_pred)