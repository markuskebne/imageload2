import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

print(tf.__version__)

SPLIT_WEIGHTS = (8,1,1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)