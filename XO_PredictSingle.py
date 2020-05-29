from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE_X = 300
IMG_SIZE_Y = 600

CLASS_LIST = ["fail", "pass"]

model = load_model('redgreen.model')
model.load_weights('redgreen.h5')

img_path=r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\Mobile2Test\72e965e4-8c63-4d95-9a49-665d5807f01d.png"

print(model.summary())

img = image.load_img(img_path, target_size=(IMG_SIZE_Y,IMG_SIZE_X))

batch_holder = np.zeros((1, IMG_SIZE_Y, IMG_SIZE_X, 3))
batch_holder[0, :] = img

result = model.predict_classes(batch_holder)

plt.imshow(img)
plt.title(CLASS_LIST[result[0][0]])
plt.show()
