import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from PIL import Image
import tensorflow as tf
import pathlib
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

AUTOTUNE = tf.data.experimental.AUTOTUNE

DATADIR = r"C:\Users\me919585\PycharmProjects\imageload\flower_photos"
CATEGORIES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMG_SIZE = 150

flowers_root = pathlib.Path(DATADIR)

# Create a list of images as a dataset
list_of_files_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

random_split()
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(filename):
    parts = tf.strings.split(filename, '\\')
    print(parts)
    label = parts[-2]
    print(label)
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

images_ds = list_of_files_ds.map(parse_image)

#data =
#labels =

#(trainX, testX, trainY, testY) = train_test_split(data,	labels, test_size=0.25, random_state=42)