import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

from PIL import Image

DATADIR = r"C:\Users\me919585\PycharmProjects\imageload\flower_photos"
CATEGORIES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMG_SIZE = 256

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])

create_training_data()

random.shuffle(training_data)

img = training_data[0]
plt.imshow(img[0], cmap="gray")

X = []
y = []

for features, label in training_data:
        X.append(features)
        y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(len(training_data), X[0].shape)

img = X[0]
print("Image shape: ", img.shape)
plt.imshow(img[0], cmap="gray")
plt.colorbar()
plt.show()


X = np.array(X) #.reshape(IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

img = X[0]
print("Image shape: ", img.shape)
plt.imshow(img[0], cmap="gray")
plt.colorbar()
plt.show()

print("Image shape: ", training_data[0][0].shape)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#X = pickle.load(open("X.pickle", "rb"))
#y = pickle.load(open("y.pickle", "rb"))
