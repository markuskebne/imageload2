from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy

Image_width = 380
Image_height = 850

CLASS_LIST = ["fail", "pass"]

model = load_model('ProductGridPage.model')
model.load_weights('ProductGridPage.h5')

print(model.summary())

image_directory=r"C:\Users\me919585\PycharmProjects\imageload\ScreenshotsSortedv2\ProductBlock\ProductGridPageTest"
numberOfFiles = sum([len(files) for r, d, files in os.walk(image_directory)])

batch_holder = np.zeros((numberOfFiles, Image_height, Image_width, 3))
for i,img in enumerate(os.listdir(image_directory)):
  img = image.load_img(os.path.join(image_directory,img), target_size=(Image_height, Image_width))
  batch_holder[i, :] = img

result = model.predict_classes(batch_holder)
probabilities = model.predict(batch_holder)

fig = plt.figure(figsize=(10, 10))

for i, img in enumerate(batch_holder):
  fig.add_subplot(2, 5, i + 1)
  prob = format(probabilities[i].max(), '7.4f')
  plt.title(CLASS_LIST[result[i]] + " " + prob)
  plt.axis('off')
  plt.imshow(img / 256.)

plt.show()