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

base_path = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\XOtest"
pic_size_X = 300
pic_size_Y = 600
batch_size = 20

## Load images from folders
## Into one training set and one validation set
datagen_test = ImageDataGenerator(validation_split=0.2)
test_generator = datagen_test.flow_from_directory(base_path,
                                                    target_size=(pic_size_Y,pic_size_X),
                                                    color_mode="rgb",
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    shuffle=True)

model = load_model('CNN.model')

epochs = 2
batch_size = 10
checkpoint = ModelCheckpoint("model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=test_generator,
                                steps_per_epoch=10,
                                epochs=epochs,
                                callbacks=callbacks_list
                                )

print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()