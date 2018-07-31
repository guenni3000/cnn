import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
import os

dir = "C:/Users/Julian/Desktop/dataset/training"
max_size = 500

def get_imgs(path):
    for file in os.listdir(os.fsencode(path)):
        filename = os.fsdecode(file)
        parts = filename.split('_')
        if parts[1] == '0':
            out = parts[2][0]
            path1 = path+'/'+filename
            path2 = path+'/'+parts[0]+'_1_'+parts[2]
            path3 = path+'/'+parts[0]+'_2_'+parts[2]
            yield ({'input1': tf.image.decode_image(tf.read_file(path1), 1), 
                    'input2': tf.image.decode_image(tf.read_file(path2), 1), 
                    'input_base': tf.image.decode_image(tf.read_file(path3), 1)}, {'output': out})

sess = tf.Session()

for img in get_imgs(dir):
    print(sess.run(tf.shape(img[0]['input1'])))
    print(sess.run(tf.shape(img[0]['input2'])))
    print(sess.run(tf.shape(img[0]['input_base'])))
    print(img[1]['output'])
    