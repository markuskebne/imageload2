import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.keras.utils.data_utils import Sequence
import PIL

import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random

import matplotlib.pyplot as plt

IMG_SIZE_X = 300
IMG_SIZE_Y = 600

CLASS_LIST = ["pass", "fail"]

model = load_model('CNN.model')
model.load_weights('model.h5')

model.summary()
print(model.layers[0].input_shape);

image_patha="ScreenshotsTest/a.png"
imga = image.load_img(image_patha, target_size=(IMG_SIZE_Y, IMG_SIZE_X))
plt.imshow(imga)
imga = np.expand_dims(imga, axis=0)
resulta=model.predict_classes(imga)
print(resulta)
plt.title(CLASS_LIST[resulta[0][0]])
plt.show()

image_pathb="ScreenshotsTest/b.png"
imgb = image.load_img(image_pathb, target_size=(IMG_SIZE_Y, IMG_SIZE_X))
plt.imshow(imgb)
imgb = np.expand_dims(imgb, axis=0)
resultb=model.predict_classes(imgb)
print(resultb)
plt.title(CLASS_LIST[resultb[0][0]])
plt.show()
