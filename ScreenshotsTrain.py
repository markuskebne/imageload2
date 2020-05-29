import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import pickle

import matplotlib.pyplot as plt

# Opening the files about data
X = pickle.load(open("Xscreens.pickle", "rb"))
y = pickle.load(open("yscreens.pickle", "rb"))

X = X/255.0
print(X.shape[1:])

# Building the model
model = tf.keras.Sequential([

	tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', input_shape=X.shape[1:]),
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
	tf.keras.layers.Dense(1, 'sigmoid')
])

# Compiling the model using some basic parameters
model.compile(loss="binary_crossentropy",
              optimizer="adam",
			  metrics=["accuracy"])
model.save_weights("model.h5")
model.save('CNN.model')


# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, y, batch_size=10, epochs=2, validation_split=0.5)
print(model.evaluate(X,y))

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model.h5")

model.save('CNN.model')

# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
