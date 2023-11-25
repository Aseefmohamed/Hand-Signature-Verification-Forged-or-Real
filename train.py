import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import cv2
import random

categories = ['Fake', 'Real']
TIME_STEPS = 31
IMG_HEIGHT = 64
IMG_WIDTH = 64

def createRCNNModel(input_shape):
    if os.path.exists('./model/model_rcnn.h5'):
        try:
            print(__name__)
            model = keras.models.load_model('./model/model_rcnn.h5')
            print("Model loaded successfully.")
            return model
        except (OSError, IOError, ValueError) as e:
            print(f"Error loading model: {e}")
    else:
        model = keras.Sequential([
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Reshape((TIME_STEPS, -1)),  # Reshape for LSTM input
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax")  # Assuming 2 classes (Fake and Real)
        ])
        return model

def getData():
    rawdata = []
    data = []
    dir = "./data/"
    for category in categories:
        path = os.path.join(dir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                rawdata = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_data = cv2.resize(rawdata, (IMG_WIDTH, IMG_HEIGHT))
                data.append([new_data, class_num])
            except Exception as e:
                pass

    random.shuffle(data)

    img_data = []
    img_labels = []
    for features, label in data:
        img_data.append(features)
        img_labels.append(label)
    img_data = np.array(img_data).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)  # Add channel dimension
    img_data = img_data / 255.0
    img_labels = np.array(img_labels)

    return img_data, img_labels

data, labels = getData()
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.20)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.10)

input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
model = createRCNNModel(input_shape)

checkpoint = ModelCheckpoint(filepath='./model/model_rcnn.h5', save_best_only=True, monitor='val_loss', mode='min')

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_data, train_labels, epochs=100, validation_data=(val_data, val_labels), callbacks=[checkpoint])

model.save('./model/model_rcnn.h5')
