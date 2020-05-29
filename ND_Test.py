from utils import *
from kh_tools import *
import ND_CreateModel
import imp
imp.reload(ND_CreateModel)
from ND_CreateModel import ALOCC_Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


adversarial_model = load_model('checkpoint\ALOCC_Model.model')
adversarial_model.load_weights('checkpoint\ALOCC_Model_7.h5')

base_path = r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock"
pic_size_X = 800
pic_size_Y = 800

colors = 3  #1 for grayscale, 3 for rgb

datagen_test = ImageDataGenerator()

test_generator = datagen_test.flow_from_directory(base_path + r"\NDTest",
                                                    target_size=(pic_size_Y, pic_size_X),
                                                    color_mode="rgb",
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    shuffle=True)

X_test, y_test = test_generator.next()
X_test = X_test / 255
data = X_test.reshape(-1, pic_size_X, pic_size_Y, colors)

adversarial_model.summary()


def test_reconstruction(data_index=0):
    #specific_idx = np.where(y_train == label)[0]
    #if data_index >= len(X_train):
    #    data_index = 0
    data = X_test.reshape(-1, pic_size_X, pic_size_Y, colors)[data_index:data_index+1]
    model_predicts = adversarial_model.predict(data)

    fig = plt.figure(figsize=(8, 8))
    columns = 1
    rows = 2
    fig.add_subplot(rows, columns, 1)
    input_image = data.reshape((pic_size_X, pic_size_Y, colors))
    reconstructed_image = model_predicts[0].reshape((pic_size_X, pic_size_Y, colors))
    plt.title('Input')
    plt.imshow(input_image, cmap='gray', label='Input')
    fig.add_subplot(rows, columns, 2)
    plt.title('Reconstruction')
    plt.imshow(reconstructed_image, cmap='gray', label='Reconstructed')
    plt.show()
    # Compute the mean binary_crossentropy loss of reconstructed image.
    y_true = K.variable(reconstructed_image)
    y_pred = K.variable(input_image)
    error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
    print('Reconstruction loss:', error)
    print('Discriminator Output:', model_predicts[1][0][0])

def test_reconstructionAlternate(data_index=0):
    # specific_idx = np.where(y_train == label)[0]
    # if data_index >= len(X_train):
    #    data_index = 0
    data = X_test.reshape(-1, pic_size_X, pic_size_Y, colors)[data_index:data_index + 1]
    model_predicts = adversarial_model.predict(data)

    fig.add_subplot(2, 6, data_index + 1 )
    input_image = data.reshape((pic_size_X, pic_size_Y, colors))
    reconstructed_image = model_predicts[0].reshape((pic_size_X, pic_size_Y, colors))

    y_true = K.variable(reconstructed_image)
    y_pred = K.variable(input_image)
    error = K.eval(binary_crossentropy(y_true, y_pred)).mean()

    plt.imshow(input_image, cmap='gray', label='Input')
#    plt.xlabel('loss' + format(error, '7.4f') + '\n' + 'output' + format(model_predicts[1][0][0], '7.4f'))
    plt.xlabel('loss' + format(error, '7.4f') + '\n' + 'output')

    print('Reconstruction loss:', error)
    print('Discriminator Output:', model_predicts[1][0][0])

for i in range(len(X_test)):
    test_reconstruction(i)

#fig = plt.figure(figsize=(10, 10))
#for i in range(len(X_test)):
#    test_reconstructionAlternate(i)

plt.show()

