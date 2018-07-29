import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import keras

dir = "C:/Users/Julian/Desktop/dataset"
max_size = 500

img = Image.open(dir+'/training1/'+os.fsdecode(os.listdir(os.fsencode(dir+'/training1/'))[0]))

height = img.height
width = img.width

ratio = (width*1.0)/height

if height > width:
    height = max_size
    width = int(max_size*ratio)
else:
    width = max_size
    height = int(max_size/ratio)

ratio = (img.width*1.0)/img.height

datagen = ImageDataGenerator()

train_iterator_base = datagen.flow_from_directory(dir+'/trainingBase/', target_size=(height, width), color_mode='grayscale', class_mode='binary', shuffle=False)
train_iterator1 = datagen.flow_from_directory(dir+'/training1/', target_size=(height, width), color_mode='grayscale', class_mode='binary', shuffle=False)
train_iterator2 = datagen.flow_from_directory(dir+'/training2/', target_size=(height, width), color_mode='grayscale', class_mode='binary', shuffle=False)

input1 = Input(shape=train_iterator1.image_shape, dtype='int32', name='input1')
input2 = Input(shape=train_iterator2.image_shape, dtype='int32', name='input2')
input_base = Input(shape=train_iterator_base.image_shape, dtype='int32', name='input_base')

conv_layer = Conv2D(16, (3, 3), padding='same', activation='relu')

conv1 = conv_layer(input1)
conv2 = conv_layer(input2)
conv_base = conv_layer(input_base)

pooling_layer = MaxPooling2D()

merged1 = keras.layers.concatenate([conv1, conv_base], axis=-1)
merged2 = keras.layers.concatenate([conv2, conv_base], axis=-1)
