import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import os, os.path

from tensorflow_core.contrib.learn.python.learn.estimators._sklearn import train_test_split

modelName = 'xo.model'
modelWeightsName = 'xo.h5'
image_directory = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\XOtrain"
Image_width = 333
Image_height = 567
batch_size = 20
epochs = 5
#color_mode = "grayscale"
color_mode = "rgb"
colors = 3

# Create Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', input_shape=(Image_height, Image_width, colors)),
    tf.keras.layers.Conv2D(kernel_size=3, filters=30, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(kernel_size=3, filters=60, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(kernel_size=3, filters=90, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(kernel_size=3, filters=110, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(kernel_size=3, filters=130, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=1, filters=40, padding='same', activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, 'softmax')
])

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.summary()
# Save model
model.save_weights(modelWeightsName)
model.save(modelName)

# Load images
datagen_train = ImageDataGenerator()
numberOfFiles = sum([len(files) for r, d, files in os.walk(image_directory)])

## Load images from folders
train_generator = datagen_train.flow_from_directory(image_directory,
                                                    target_size=(Image_height, Image_width),
                                                    color_mode=color_mode,
                                                    batch_size=numberOfFiles,
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    seed=42)

#Split the data into training and validation sets
X_train, y_train = train_generator.next()
X_train, X_validate, y_train, y_validate = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.3,
                                                            random_state=42)

model = load_model(modelName)
model.summary()

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

checkpoint = ModelCheckpoint(modelWeightsName, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train the model
history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    class_weight=class_weights,
                    callbacks=callbacks_list,
                    validation_data=(X_validate, y_validate)
                    )

# Plot results from training
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

