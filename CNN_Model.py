# Libraries
import numpy as np
import pandas as pd

import cv2
import os

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import json
import h5py

# Set the path to the data folder
data_path = "dataset/grayscale_images"

# Set the image size and batch size
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Create data generators for train and test data
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    data_path + "/train",
    target_size=img_size,
    batch_size=batch_size,
    color_mode = "grayscale",
    class_mode="sparse",
    # shuffle=True
)

test_data = test_datagen.flow_from_directory(
    data_path + "/test",
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode="sparse",
    # classes = dir_names,
    shuffle=False
)

# CNN Model Architecture

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_size[0], img_size[1], 1), padding="same"),
    MaxPooling2D((2, 2), padding="valid"),
    # Dropout(0.25),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2), padding="valid"),
    # Dropout(0.25),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2), padding="valid"),
    # Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.40),
    Dense(96, activation="relu"),
    Dropout(0.40),
    Dense(64, activation="relu"),
    Dense(36, activation="softmax") # softmax for more than 2
])

save_best_cb = tf.keras.callbacks.ModelCheckpoint('models/initial-end-to-end', save_best_only = True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 5)

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", "categorical_accuracy", "sparse_categorical_accuracy", "top_k_categorical_accuracy"]
)

# Fitting the Model
model.fit(
    train_data,
    # steps_per_epoch=steps_per_epoch_train,
    epochs=20,
    validation_data=test_data
    # validation_steps=steps_per_epoch_test
    )

# Saving the model and its weights
model_json = model.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
model.save_weights('model_new.h5')
print('Weights saved')

# Loading model from dir

# Load the JSON file that contains the model architecture
with open('model_new.json', 'r') as json_file:
    model_json = json_file.read()

# Load the model architecture from the JSON file
loaded_model = tf.keras.models.model_from_json(model_json)

# Load the saved weights into the model
loaded_model.load_weights('model_new.h5')

# Compile the loaded model
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Use the loaded model for prediction or evaluation

test_loss, test_acc = loaded_model.evaluate(test_data)

print("Test accuracy:", test_acc)
