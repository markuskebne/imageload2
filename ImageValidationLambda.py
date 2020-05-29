from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

img_path=r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\ProductGridPageTest\ProductGridPage_0e74d581-b638-4fc3-a4a0-92b0d127aa47.jpg"

IMG_SIZE_X = 380
IMG_SIZE_Y = 850

CLASS_LIST = ["fail", "pass"]

model = load_model('ProductGridPage.model')
model.load_weights('ProductGridPage.h5')

img = image.load_img(img_path, target_size=(IMG_SIZE_Y,IMG_SIZE_X))
img = np.resize(img, (1, IMG_SIZE_Y, IMG_SIZE_X, 3))

prediction = model.predict(img)

predicted_class = CLASS_LIST[prediction.argmax(axis=-1)[0]]
probability = format(prediction.max(), '7.4f')

print("hej")

