import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
import os

dir = "C:/Users/Julian/Desktop/dataset/training"
max_size = 500

filt = [0]

add = [3, 6]

arr = np.array(filt, np.int8)
arr_add = np.array(add, np.int8)

arr[0] = arr_add

sess = tf.Session()

print(sess.run(tf.shape(arr)))
    