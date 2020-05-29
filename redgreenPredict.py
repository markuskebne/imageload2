from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy

Image_width = 300
Image_height = 600

CLASS_LIST = ["green", "red"]

model = load_model('redgreen.model')
model.load_weights('redgreen.h5')

print(model.summary())

batch_holder = np.zeros((10, Image_height, Image_width, 3))
img_dir=r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\Mobile2Test"
for i,img in enumerate(os.listdir(img_dir)):
  img = image.load_img(os.path.join(img_dir,img), target_size=(Image_height, Image_width))
  batch_holder[i, :] = img

result = model.predict_classes(batch_holder)

fig = plt.figure(figsize=(10, 10))

for i, img in enumerate(batch_holder):
  fig.add_subplot(4, 5, i + 1)
  plt.title(CLASS_LIST[result[i]])
  plt.axis('off')
  plt.imshow(img / 256.)

plt.show()