import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

#########
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

base_path = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock"
pic_size_X = 300
pic_size_Y = 600
batch_size = 10

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

train_generator = datagen_train.flow_from_directory(base_path + "\ColorTrain",
                                                    target_size=(pic_size_Y,pic_size_X),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    shuffle=True)

validation_generator = datagen_validation.flow_from_directory(base_path + "\ColorValidation",
                                                    target_size=(pic_size_Y,pic_size_X),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    shuffle=False)

model = load_model('CNN.model')

epochs = 3
checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=16,
                                epochs=epochs,
                                validation_data=validation_generator,
                                validation_steps=16,
                                callbacks=callbacks_list
                                )










########
file_list = []
class_list = []

DATADIR = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\Mobile"
CATEGORIES = ["pass", "fail"]

IMG_SIZE_X = 300
IMG_SIZE_Y = 600

for category in CATEGORIES:
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img))

training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img))
				new_array = cv2.resize(img_array, (IMG_SIZE_X, IMG_SIZE_Y))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()
random.shuffle(training_data)

img = training_data[0]
plt.imshow(img[0])
plt.colorbar()
plt.show()

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 3)
plt.imshow(X[0])
plt.colorbar()
plt.show()

# Creating the files containing all the information about your model
pickle_out = open("Xscreens.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("yscreens.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()