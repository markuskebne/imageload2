import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print(X[0].shape)
print(X.shape[1:])

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=32, epochs=1, validation_split=0.1)


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle