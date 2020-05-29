from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import tensorflow as tf

base_path = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock"
pic_size_X = 28 #300
pic_size_Y = 28 #600
batch_size = 20

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

## Load images from folders
## Into one training set and one validation set
train_generator = datagen_train.flow_from_directory(base_path + r"\NDTrain",
                                                    target_size=(pic_size_Y,pic_size_X),
                                                    color_mode="grayscale",
                                                    batch_size=487,
                                                    class_mode='binary',
                                                    shuffle=True)

X_train, y_train = train_generator.next()


validation_generator = datagen_validation.flow_from_directory(base_path + r"\NDValidate",
                                                    target_size=(pic_size_Y,pic_size_X),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    shuffle=False)

model = load_model('ALOCC_Model.model')
model.summary()

epochs = 3

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}



checkpoint = ModelCheckpoint("ALOCC_Model_checkpoint.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_generator,
                                #steps_per_epoch=train_generator.__len__() / batch_size,
                                epochs=epochs,
                                validation_data=validation_generator,
                                #validation_steps=validation_generator.__len__() / batch_size,
                                callbacks=callbacks_list,
                                class_weight=class_weights
                                )

model.evaluate_generator(generator=validation_generator)

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