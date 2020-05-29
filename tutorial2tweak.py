import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
import pickle
import cv2

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

print(X[0].shape)
print(X.shape[1:])

X = X/255.0
img = X[0]
plt.imshow(img[0])
plt.colorbar()
plt.show()


model = Sequential()
#model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
#model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Flatten(input_shape=X.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
#model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#print(model.summary())
model.fit(X, y, batch_size=32, epochs=1, validation_split=0.1)


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle