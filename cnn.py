import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import keras

dir = "C:/Users/Julian/Desktop/dataset"
max_size = 500

def compare(x):
    img1 = x[0:length(x)]

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

pooling_layer = MaxPooling2D(pool_size=(4, 4))

pool1 = pooling_layer(conv1)
pool2 = pooling_layer(conv2)
pool_base = pooling_layer(conv_base)

merged1 = keras.layers.concatenate([pool1, pool_base], axis=-1)
merged2 = keras.layers.concatenate([pool2, pool_base], axis=-1)

merged1 = Dropout(0.25)(merged1)
merged2 = Dropout(0.25)(merged2)

compare_layer = Lambda(compare)

comp1 = compare_layer(merged1)
comp2 = compare_layer(merged2)

comp1 = pooling_layer(comp1)
comp2 = pooling_layer(comp2)

comp1 = Flatten()(comp1)
comp2 = Flatten()(comp2)

merged_final = keras.layers.concatenate([comp1, comp2], axis=-1)

x = Dense(64, activation='relu')(merged_final)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[input1, input2], outputs=[output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit({'input1': train_iterator1, 'input2': train_iterator2, 'input_base': train_iterator_base},
          {'output': train_iterator_base},
          epochs=10, batch_size=32)
