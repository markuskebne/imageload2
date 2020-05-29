from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from keras.utils import to_categorical
import tensorflow as tf
import os, os.path

from tensorflow_core.contrib.learn.python.learn.estimators._sklearn import train_test_split

image_directory = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\Mobile2Train"
base_path = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock"
modelName = 'redgreen.model'
modelWeightsName = 'redgreen.h5'
Image_width = 300
Image_height = 600
batch_size = 20
epochs = 20
#color_mode = "grayscale"
color_mode = "rgb"

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

numberOfFiles = sum([len(files) for r, d, files in os.walk(image_directory)])

## Load images from folders
## Into one training set and one validation set
train_generator = datagen_train.flow_from_directory(image_directory,
                                                    target_size=(Image_height, Image_width),
                                                    color_mode=color_mode,
                                                    batch_size=numberOfFiles,
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    seed=42)

X_train, y_train = train_generator.next()
X_train, X_validate, y_train, y_validate = train_test_split(X_train,
                                                    y_train,
                                                    test_size=0.3,
                                                    random_state=42)

#validation_generator = datagen_validation.flow_from_directory(base_path + "\Mobile2Validate",
#                                                    target_size=(Image_height, Image_width),
#                                                    color_mode="rgb",
#                                                    batch_size=1,
#                                                    class_mode='binary',
#                                                    shuffle=True,
#                                                    seed=42)

model = load_model(modelName)
model.summary()

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

checkpoint = ModelCheckpoint(modelWeightsName, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                class_weight=class_weights,
                callbacks=callbacks_list,
                validation_data=(X_validate, y_validate)
               )

#history = model.fit_generator(generator=train_generator,
#                                steps_per_epoch=train_generator.__len__() / train_generator.batch_size,
#                                epochs=epochs,
#                                validation_data=validation_generator,
#                                validation_steps=validation_generator.__len__() / validation_generator.batch_size,
#                                callbacks=callbacks_list,
#                                class_weight=class_weights
#                                )

#model.evaluate_generator(generator=validation_generator)

print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'train_loss', 'val-loss'], loc='upper left')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()