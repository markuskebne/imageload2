from tensorflow.keras.models import load_model
#import keras.backend as ksbe
from keras.preprocessing import image
import numpy as np
#import matplotlib.pyplot as plt

IMG_SIZE_X = 380
IMG_SIZE_Y = 850

CLASS_LIST = ["fail", "pass"]

model = load_model('ProductGridPage.model')
model.load_weights('ProductGridPage.h5')

img_path=r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\ProductGridPageTest\Clicking_customer_service_link_should_open_customer_service_page_1fc556af-f2d6-4ad5-8ca5-580e3ff19a02.jpg"

img = image.load_img(img_path, target_size=(IMG_SIZE_Y,IMG_SIZE_X))

#img2 = ksbe.reshape(img, (1, IMG_SIZE_Y, IMG_SIZE_X, 3))

batch_holder = np.zeros((1, IMG_SIZE_Y, IMG_SIZE_X, 3))
batch_holder[0, :] = img

result = model.predict_classes(batch_holder)
probabilities = model.predict(batch_holder)

prob = format(probabilities[0].max(), '7.4f')
print(CLASS_LIST[result[0]] + " " + prob)